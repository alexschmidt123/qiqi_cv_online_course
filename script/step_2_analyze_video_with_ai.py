#!/usr/bin/env python3
"""Step 2: build an AI-defined layout library and per-video report.

OpenCV selects representative visual states and removes the red gaze marker.
EasyOCR supplies text/bounding-box evidence. The AI determines whether the video
is an article or slides, reconstructs article text, and defines semantic element
polygons. Each distinct slide/popup state receives its own annotated image.
"""
from __future__ import annotations

import argparse
import base64
import csv
import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

try:
    import easyocr
except ImportError:
    easyocr = None

DEFAULT_MODEL = "gpt-4o-mini"
COLORS = {
    "web_navigation": (255, 80, 40),
    "web_panel": (255, 180, 40),
    "webpage_title": (30, 80, 255),
    "slide_title": (30, 80, 255),
    "paragraph": (40, 180, 40),
    "image": (220, 70, 220),
    "button": (0, 170, 255),
    "popup": (180, 40, 255),
    "slide_navigation": (255, 150, 0),
    "blank_area": (150, 150, 150),
}


@dataclass
class Sample:
    sample_id: str
    pattern_id: str
    timestamp: float
    frame: np.ndarray
    clean_frame: np.ndarray
    ocr: List[Dict[str, Any]]
    gaze: Optional[Tuple[float, float]] = None


def _rect_polygon(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _locate_content_roi(frame: np.ndarray, video_type: str) -> List[List[float]]:
    """Locate the course content before state comparison or semantic analysis.

    This course template has a long pink control strip immediately below the
    content. OpenCV uses that strip to obtain the horizontal bounds and content
    bottom. The article then uses the inset scrollable reading pane.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink = cv2.inRange(hsv, np.array([145, 35, 80]), np.array([179, 255, 255]))
    best: Optional[Tuple[int, int, int]] = None
    for y in range(int(0.48 * h), int(0.86 * h)):
        xs = np.flatnonzero(pink[y] > 0)
        if len(xs) < int(0.20 * w):
            continue
        x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        if x1 - x0 >= int(0.35 * w) and (best is None or x1 - x0 > best[2] - best[1]):
            best = (y, x0, x1)
    if best:
        bottom, x0, x1 = best
        # Pink rows can span several pixels; their first row is the content edge.
        while bottom > int(0.35 * h) and np.count_nonzero(pink[bottom - 1, x0:x1]) > 0.35 * (x1 - x0):
            bottom -= 1
        # The colored strip has inset controls, so extend its measured run to
        # the outer white slide edges before deriving the aspect ratio.
        inset = round(.02 * (x1 - x0))
        x0, x1 = max(0, x0 - inset), min(w, x1 + inset)
        # The player content uses a stable 16:8.45 viewport above the controls.
        top = max(0, round(bottom - 0.529 * (x1 - x0)))
    else:
        x0, x1, top, bottom = round(.222 * w), round(.778 * w), round(.174 * h), round(.697 * h)
    if video_type == "article":
        # Only the scrolling white reading pane is article content. The course
        # title and player navigation remain unrelated even though inside player.
        span_x, span_y = x1 - x0, bottom - top
        x0, x1 = round(x0 + .09 * span_x), round(x1 - .09 * span_x)
        top, bottom = round(top + .265 * span_y), round(bottom - .055 * span_y)
    return _rect_polygon(x0 / w, top / h, x1 / w, bottom / h)


def _crop_to_roi(frame: np.ndarray, roi: Sequence[Sequence[float]]) -> np.ndarray:
    h, w = frame.shape[:2]
    xs, ys = [p[0] for p in roi], [p[1] for p in roi]
    x0, x1 = max(0, round(min(xs) * w)), min(w, round(max(xs) * w))
    y0, y1 = max(0, round(min(ys) * h)), min(h, round(max(ys) * h))
    return frame[y0:y1, x0:x1]


def _remove_red_gaze(frame: np.ndarray) -> np.ndarray:
    """Remove the red gaze marker from screenshots before state comparison/AI."""
    out = frame.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 115, 70]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([168, 115, 70]), np.array([180, 255, 255])),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    gaze_mask = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        ratio = min(w, h) / max(w, h) if max(w, h) else 0
        if 25 <= area <= 3500 and ratio >= 0.45:
            gaze_mask[labels == i] = 255
    if np.any(gaze_mask):
        gaze_mask = cv2.dilate(gaze_mask, np.ones((7, 7), np.uint8), iterations=1)
        out = cv2.inpaint(out, gaze_mask, 5, cv2.INPAINT_TELEA)
    return out


def _visual_difference(a: np.ndarray, b: np.ndarray, roi: Sequence[Sequence[float]]) -> float:
    def prep(frame: np.ndarray) -> np.ndarray:
        frame = _crop_to_roi(frame, roi)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(prep(a), prep(b))))


def _ocr_around_gaze(
    reader: Any,
    frame: np.ndarray,
    width: int,
    height: int,
    gaze: Optional[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """OCR only the article text surrounding the gaze point, never the full frame."""
    if gaze is None:
        return []
    if reader is None:
        return []
    gx, gy_tl = gaze
    x0, x1 = max(0, int(gx) - 420), min(width, int(gx) + 420)
    y0, y1 = max(0, int(gy_tl) - 240), min(height, int(gy_tl) + 240)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    rows = []
    for bbox, text, confidence in reader.readtext(crop):
        if not str(text).strip() or float(confidence or 0) < 0.08:
            continue
        points = [[round((float(x) + x0) / width, 5), round((float(y) + y0) / height, 5)] for x, y in bbox]
        rows.append({"text": str(text).strip(), "confidence": round(float(confidence), 3), "polygon": points})
    return rows


def _load_gaze_by_time(path: Path, height: int) -> List[Tuple[float, float, float]]:
    points = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                ts, x, y_bl = float(row["timestamp_sec"]), float(row["x_bl"]), float(row["y_bl"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((ts, x, (height - 1) - y_bl))
    return points


def _nearest_gaze(points: Sequence[Tuple[float, float, float]], timestamp: float, tolerance: float) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    ts, x, y = min(points, key=lambda p: abs(p[0] - timestamp))
    return (x, y) if abs(ts - timestamp) <= tolerance else None


def collect_samples(
    video: Path,
    gaze_csv: Path,
    video_type: str,
    interval: float,
    change_threshold: float,
    max_samples: int,
) -> Tuple[int, int, float, List[Sample], List[List[float]]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps if count else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, roi_frame = cap.read()
    if not ok or roi_frame is None:
        raise RuntimeError("Could not read a frame for content ROI detection")
    course_roi = _locate_content_roi(roi_frame, video_type)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False) if video_type == "article" and easyocr is not None else None
    gaze_points = _load_gaze_by_time(gaze_csv, height) if video_type == "article" else []
    samples: List[Sample] = []
    unique_patterns: List[Tuple[str, np.ndarray]] = []
    previous_pattern = ""
    timestamps = np.arange(0.0, max(duration, interval), interval).tolist()
    if video_type == "article" and len(timestamps) > max_samples:
        timestamps = np.linspace(0.0, max(0.0, duration - 1.0 / fps), max_samples).tolist()
    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        clean = _remove_red_gaze(frame)
        sample_id = f"sample_{len(samples) + 1:03d}"
        if video_type == "slides":
            matches = [(diff, pid) for pid, representative in unique_patterns for diff in [_visual_difference(representative, clean, course_roi)]]
            best = min(matches) if matches else None
            if best is not None and best[0] < change_threshold:
                pattern_id = best[1]
            elif len(unique_patterns) < max_samples:
                pattern_id = f"pattern_{len(unique_patterns) + 1:03d}"
                unique_patterns.append((pattern_id, clean.copy()))
            else:
                pattern_id = best[1] if best else "pattern_001"
            # Record every transition, including revisits and out-of-order navigation.
            if pattern_id == previous_pattern:
                continue
            previous_pattern = pattern_id
            samples.append(Sample(sample_id, pattern_id, float(timestamp), frame, clean, [], None))
        else:
            gaze = _nearest_gaze(gaze_points, float(timestamp), max(interval, 0.5))
            ocr = _ocr_around_gaze(reader, clean, width, height, gaze)
            samples.append(Sample(sample_id, sample_id, float(timestamp), frame, clean, ocr, gaze))
    cap.release()
    if not samples:
        raise RuntimeError("No representative frames could be read")
    return width, height, duration, samples, course_roi


def _jpeg_data_url(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("Could not encode representative frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _extract_json(text: str) -> Dict[str, Any]:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("AI response did not contain a JSON object")
        return json.loads(value[start : end + 1])


def _select_ai_screenshots(samples: Sequence[Sample], limit: int) -> List[Sample]:
    """Choose a small, time-distributed set; AI never receives every sampled frame."""
    if limit <= 0:
        raise ValueError("ai screenshot limit must be positive")
    if len(samples) <= limit:
        return list(samples)
    indices = np.linspace(0, len(samples) - 1, limit, dtype=int).tolist()
    return [samples[i] for i in sorted(set(indices))]


def analyze_with_ai(
    video_name: str,
    samples: Sequence[Sample],
    model: str,
    requested_type: str,
    ai_screenshot_limit: int,
    course_roi: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for step 2 layout analysis")
    if requested_type == "slides":
        # One screenshot for every globally unique base/popup state. Do not cap at six.
        by_pattern: Dict[str, Sample] = {}
        for sample in samples:
            by_pattern.setdefault(sample.pattern_id, sample)
        representatives = list(by_pattern.values())
        if len(representatives) > 6:
            merged: Dict[str, Any] = {"video_type": "slides", "summary": "", "article": {}, "states": []}
            summaries = []
            for start in range(0, len(representatives), 6):
                batch = representatives[start : start + 6]
                part = analyze_with_ai(video_name, batch, model, "slides", 6, course_roi)
                batch_states = list(part.get("states", []))
                for index, state in enumerate(batch_states[: len(batch)]):
                    # Model-generated IDs are local to each request. OpenCV pattern IDs
                    # are global across the video and remain stable on slide revisits.
                    state["sample_id"] = batch[index].pattern_id
                    state["state_id"] = batch[index].pattern_id
                    state["content_id"] = batch[index].pattern_id
                merged["states"].extend(batch_states)
                if part.get("summary"):
                    summaries.append(str(part["summary"]))
            merged["summary"] = " ".join(summaries)
            return merged
    else:
        representatives = _select_ai_screenshots(samples, ai_screenshot_limit)
    representative_ids = {s.sample_id for s in representatives}
    # OCR is local and inexpensive. Keep it for all states so article text and slide
    # labels are not lost, but send pixels for only a few pattern examples.
    evidence = {
        s.sample_id: {
            "timestamp_sec": round(s.timestamp, 3),
            "pattern_id": s.pattern_id,
            "has_screenshot": s.sample_id in representative_ids,
            "ocr": s.ocr,
        }
        for s in samples
    }
    prompt = f"""
Analyze cropped online-course content screenshots from the video {video_name!r}.
Requested video type: {requested_type}. If it is auto, classify it as exactly "article" or "slides".

OCR evidence (normalized top-left coordinates) follows. Correct OCR mistakes using the images:
{json.dumps(evidence, ensure_ascii=False)}

For article videos, only a few representative screenshots are attached and OCR contains only text
around gaze points. Reconstruct the full article from those gaze-centered observations.
For slide videos, OCR is intentionally empty and one screenshot is attached for every globally
unique pattern_id, including popup states. Navigation may be out of order. Repeated visits reuse
the same pattern_id. Define one semantic state per pattern_id.

Return JSON only with this structure:
{{
  "video_type": "article|slides",
  "summary": "short description",
  "article": {{
    "title": "",
    "full_text": "complete article in reading order, empty for slides",
    "layout_labels": [{{"label":"heading|paragraph|image|navigation|panel", "text":"", "sample_id":"sample_001"}}]
  }},
  "states": [
    {{
      "sample_id": "sample_001",
      "content_id": "article id for articles, stable logical slide id for slides",
      "state_id": "unique id; popup/click result must be a separate state",
      "state_description": "",
      "elements": [
        {{
          "element_id": "stable unique id",
          "element_type": "web_navigation|web_panel|webpage_title|slide_title|paragraph|image|button|popup|slide_navigation|blank_area",
          "label": "specific human-readable attention content, not merely the type",
          "priority": 0,
          "polygons": [[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]]
        }}
      ]
    }}
  ]
}}

Rules:
- Every polygon uses normalized top-left coordinates within the attached cropped content image.
- Use multiple polygons for disconnected regions.
- The attachment itself is the content ROI; do not invent surrounding webpage elements.
- For article videos, the crop is a fixed scrollable viewport and no single screenshot contains the
  full article. Reconstruct the article across time-ordered gaze OCR, remove overlap, and preserve order.
- For slides, include every distinct slide state. A popup revealed by a clickable button is a new
  state even when the base slide is unchanged. Each screenshot contains the complete slide inside
  content crop. Give popup elements higher priority than covered content.
- Draw a tight polygon around each actual visual element. Never use the whole crop for a title,
  paragraph, image, button, or popup. The whole-crop polygon is allowed only for blank_area.
- Keep title, body paragraph, figure, each button, and each visible popup as separate elements.
  A popup polygon must tightly cover the popup panel and use element_type popup with priority >= 100.
- Do not treat the red gaze marker as content.
- For articles, produce one state for each sample_id and assign its article section.
- For slides, produce one state for each unique pattern_id and put that pattern_id in sample_id.
  Reuse slide_id for popup variants of the same base slide.
"""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for sample in representatives:
        identity = sample.pattern_id if requested_type == "slides" else sample.sample_id
        content.append({"type": "input_text", "text": f"{identity} at first occurrence {sample.timestamp:.3f}s"})
        content.append({"type": "input_image", "image_url": _jpeg_data_url(_crop_to_roi(sample.clean_frame, course_roi)), "detail": "high"})
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=12000,
    )
    return _extract_json(response.output_text)


def _normalize_polygon(raw: Any) -> List[List[float]]:
    points = []
    if not isinstance(raw, list):
        return points
    for point in raw:
        if isinstance(point, list) and len(point) >= 2:
            points.append([max(0.0, min(1.0, float(point[0]))), max(0.0, min(1.0, float(point[1])))])
    return points if len(points) >= 3 else []


def _article_section_assignments(result: Dict[str, Any], samples: Sequence[Sample]) -> Dict[str, str]:
    article = result.get("article") or {}
    full_text = str(article.get("full_text") or "")
    normalized_full = " ".join(full_text.lower().split())
    title = str(article.get("title") or "article title")
    sections: List[Tuple[float, str]] = []
    offset = 0
    for line in full_text.splitlines(True):
        label = line.strip().rstrip(":")
        letters = [c for c in label if c.isalpha()]
        if len(letters) >= 3 and (line.strip().endswith(":") or all(c.isupper() for c in letters)):
            sections.append((offset / max(1, len(full_text)), label))
        offset += len(line)
    assignments, previous = {}, title
    for sample in samples:
        positions = []
        for item in sample.ocr:
            observed = " ".join(str(item.get("text") or "").lower().split())
            if len(observed) < 8 or not normalized_full:
                continue
            match = difflib.SequenceMatcher(None, normalized_full, observed, autojunk=False).find_longest_match()
            if match.size >= min(12, max(8, len(observed) // 2)):
                positions.append(match.a / max(1, len(normalized_full)))
        if positions:
            fraction = float(np.median(positions))
            # Pure NLP decision: locate gaze-near OCR words in the reconstructed
            # article and select the preceding semantic heading. No image position
            # or scroll direction participates in this decision.
            candidates = [label for position, label in sections if position <= fraction]
            previous = candidates[-1] if candidates else title
        assignments[sample.sample_id] = previous
    return assignments


def _from_roi_polygon(polygon: Sequence[Sequence[float]], roi: Sequence[Sequence[float]]) -> List[List[float]]:
    xs, ys = [p[0] for p in roi], [p[1] for p in roi]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [[x0 + float(x) * (x1 - x0), y0 + float(y) * (y1 - y0)] for x, y in polygon]


def build_library(result: Dict[str, Any], samples: Sequence[Sample], width: int, height: int, duration: float,
                  detected_roi: Sequence[Sequence[float]]) -> Dict[str, Any]:
    by_id = {str(s.get("sample_id")): s for s in result.get("states", []) if isinstance(s, dict)}
    article_sections = _article_section_assignments(result, samples) if result.get("video_type") == "article" else {}
    states = []
    for index, sample in enumerate(samples):
        raw = by_id.get(sample.pattern_id) or by_id.get(sample.sample_id, {})
        # ROI is determined by OpenCV before AI. Never let semantic output widen it.
        course_roi = [list(map(list, detected_roi))]
        end = samples[index + 1].timestamp if index + 1 < len(samples) else duration + 1e-6
        elements = []
        for number, element in enumerate(raw.get("elements", []), 1):
            polygons = [_normalize_polygon(p) for p in element.get("polygons", [])]
            polygons = [p for p in polygons if p]
            polygons = [_from_roi_polygon(p, detected_roi) for p in polygons]
            if not polygons:
                continue
            element_type = str(element.get("element_type") or "blank_area")
            elements.append({
                "element_id": str(element.get("element_id") or f"{sample.sample_id}_element_{number:02d}"),
                "element_type": element_type,
                "label": str(element.get("label") or element.get("element_type") or "unknown"),
                "priority": int(element.get("priority") or 0),
                "polygons": polygons,
            })
        if result.get("video_type") == "article":
            # Geometry only determines inside/outside the article. The section is
            # inferred exclusively by matching gaze-local OCR to the full article.
            elements = [{
                "element_id": f"{sample.sample_id}_nlp_section",
                "element_type": "article_section",
                "label": article_sections.get(sample.sample_id, "article"),
                "priority": 10,
                "polygons": course_roi,
            }]
        elif not any(element.get("element_type") == "blank_area" for element in elements):
            # Explicit fallback makes validation distinguish background attention
            # from a failed/missing decision. Semantic elements have higher priority.
            elements.append({
                "element_id": f"{sample.pattern_id}_blank_area",
                "element_type": "blank_area",
                "label": "blank area",
                "priority": -100,
                "polygons": course_roi,
            })
        states.append({
            "sample_id": sample.sample_id,
            "pattern_id": sample.pattern_id,
            "content_id": str(raw.get("content_id") or raw.get("slide_id") or ("article" if result.get("video_type") == "article" else sample.pattern_id)),
            "state_id": str(raw.get("state_id") or sample.sample_id),
            "state_description": str(raw.get("state_description") or ""),
            "course_roi": course_roi,
            "start_sec": sample.timestamp,
            "end_sec": max(sample.timestamp + 1e-6, end),
            "elements": elements,
        })
    return {
        "schema_version": 1,
        "video_type": result.get("video_type", "unknown"),
        "frame_size": {"width": width, "height": height},
        "coordinate_system": "normalized-top-left",
        "states": states,
    }


def render_layout_images(samples: Sequence[Sample], library: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_by_id = {s.sample_id: s for s in samples}
    rendered = set()
    for state in library["states"]:
        sample = sample_by_id.get(state["sample_id"])
        if sample is None:
            continue
        if state["state_id"] in rendered:
            continue
        rendered.add(state["state_id"])
        h, w = sample.clean_frame.shape[:2]
        canvas = cv2.addWeighted(sample.clean_frame, 0.5, np.full_like(sample.clean_frame, 255), 0.5, 0)
        for element in state["elements"]:
            color = COLORS.get(element["element_type"], (0, 0, 0))
            for polygon in element["polygons"]:
                points = np.array([[round(x * w), round(y * h)] for x, y in polygon], dtype=np.int32)
                cv2.polylines(canvas, [points], True, color, 4, cv2.LINE_AA)
                x, y = points[0]
                cv2.putText(canvas, element["label"][:50], (int(x), max(18, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"{state['state_id']}.png"), canvas)


def write_report(result: Dict[str, Any], library: Dict[str, Any], output_dir: Path) -> None:
    (output_dir / "ai_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    article = result.get("article") or {}
    lines = [f"# AI Video Report", "", f"Video type: {result.get('video_type', 'unknown')}", "", str(result.get("summary", "")), ""]
    if result.get("video_type") == "article":
        (output_dir / "full_article.txt").write_text(str(article.get("full_text", "")).strip() + "\n", encoding="utf-8")
        lines += ["## Article", "", f"Title: {article.get('title', '')}", "", "### Full text", "", str(article.get("full_text", "")), "", "### Layout labels", ""]
        for item in article.get("layout_labels", []):
            lines.append(f"- {item.get('label', '')}: {item.get('text', '')} ({item.get('sample_id', '')})")
    else:
        lines += ["## Slide states", ""]
        for state in library["states"]:
            lines.append(f"- {state['state_id']} — {state['state_description']} (`layout/{state['state_id']}.png`)")
    (output_dir / "ai_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _crop_difference(a: np.ndarray, b: np.ndarray) -> float:
    def prep(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(prep(a), prep(b))))


def analyze_slides_with_standard_library(
    video_name: str, samples: Sequence[Sample], model: str, course_roi: Sequence[Sequence[float]],
    standard_dir: Path, match_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Match participant states to one course-level slide library, extending it only when needed."""
    standard_dir.mkdir(parents=True, exist_ok=True)
    template_dir = standard_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = standard_dir / "standard_library.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"states": []}
    templates = []
    for state in catalog.get("states", []):
        image = cv2.imread(str(template_dir / state["template"]))
        if image is not None:
            templates.append((state, image))

    by_pattern: Dict[str, Sample] = {}
    for sample in samples:
        by_pattern.setdefault(sample.pattern_id, sample)
    matched: Dict[str, Dict[str, Any]] = {}
    unmatched: List[Sample] = []
    for pattern_id, sample in by_pattern.items():
        crop = _crop_to_roi(sample.clean_frame, course_roi)
        candidates = [(_crop_difference(crop, image), state) for state, image in templates]
        best = min(candidates, key=lambda item: item[0]) if candidates else None
        if best and best[0] <= match_threshold:
            matched[pattern_id] = best[1]
        else:
            unmatched.append(sample)

    if unmatched:
        learned = analyze_with_ai(video_name, unmatched, model, "slides", 6, course_roi)
        learned_by_id = {str(s.get("sample_id")): s for s in learned.get("states", [])}
        next_number = len(catalog.get("states", [])) + 1
        for sample in unmatched:
            raw = learned_by_id.get(sample.pattern_id, {})
            standard_id = f"slide_{next_number:03d}"
            filename = f"{standard_id}.png"
            cv2.imwrite(str(template_dir / filename), _crop_to_roi(sample.clean_frame, course_roi))
            standard_state = {
                "standard_id": standard_id,
                "content_id": standard_id,
                "state_description": str(raw.get("state_description") or ""),
                "elements": list(raw.get("elements") or []),
                "template": filename,
            }
            catalog.setdefault("states", []).append(standard_state)
            templates.append((standard_state, cv2.imread(str(template_dir / filename))))
            matched[sample.pattern_id] = standard_state
            next_number += 1
        catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    states = []
    for pattern_id in by_pattern:
        standard = matched[pattern_id]
        states.append({
            "sample_id": pattern_id,
            "content_id": standard["content_id"],
            "state_id": standard["standard_id"],
            "state_description": standard.get("state_description", ""),
            "elements": standard.get("elements", []),
        })
    return {"video_type": "slides", "summary": "States matched to the shared course slide library.", "article": {}, "states": states}


def main() -> None:
    parser = argparse.ArgumentParser(description="AI video classification, semantic layout library, and report")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gaze-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--standard-library-dir", type=Path)
    parser.add_argument("--type", choices=("auto", "article", "slides"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.7,
        help="OpenCV slide-state deduplication threshold; low enough to retain popup changes.",
    )
    parser.add_argument("--max-samples", type=int, default=80, help="Maximum article observations or unique slide patterns")
    parser.add_argument(
        "--ai-screenshots",
        type=int,
        default=6,
        help="Maximum representative screenshots sent to AI; OCR may cover more local samples.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, height, duration, samples, course_roi = collect_samples(
        args.video, args.gaze_csv, args.type, args.sample_interval, args.change_threshold, args.max_samples
    )
    if args.type == "slides" and args.standard_library_dir:
        result = analyze_slides_with_standard_library(
            args.video.name, samples, args.model, course_roi, args.standard_library_dir
        )
    else:
        result = analyze_with_ai(args.video.name, samples, args.model, args.type, args.ai_screenshots, course_roi)
    if args.type != "auto":
        result["video_type"] = args.type
    library = build_library(result, samples, width, height, duration, course_roi)
    (args.output_dir / "element_library.json").write_text(json.dumps(library, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_layout_images(samples, library, args.output_dir / "layout")
    write_report(result, library, args.output_dir)
    print(f"Wrote {args.output_dir / 'element_library.json'} and AI report")


if __name__ == "__main__":
    main()
