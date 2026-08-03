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
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _display_text(value: str) -> str:
    value = value.replace("’", "'").replace("“", '"').replace("”", '"').replace("…", "...")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


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


def _ellipse_polygon(cx: float, cy: float, rx: float, ry: float, points: int = 24) -> List[List[float]]:
    return [[cx + rx * float(np.cos(2 * np.pi * i / points)), cy + ry * float(np.sin(2 * np.pi * i / points))] for i in range(points)]


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


def _remove_red_gaze(frame: np.ndarray, gaze: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remove the red gaze marker from screenshots before state comparison/AI."""
    out = frame.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 115, 70]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([168, 115, 70]), np.array([180, 255, 255])),
    )
    gaze_mask = np.zeros(mask.shape, dtype=np.uint8)
    if gaze is not None:
        # Step 1 already located the gaze marker. Restrict removal to that
        # neighborhood so pink/red slide titles and diagram content survive.
        local = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(local, (round(gaze[0]), round(gaze[1])), 34, 255, -1)
        gaze_mask = cv2.bitwise_and(mask, local)
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
    gaze_points = _load_gaze_by_time(gaze_csv, height) if gaze_csv.exists() else []
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
        gaze = _nearest_gaze(gaze_points, float(timestamp), max(interval, 0.5))
        clean = _remove_red_gaze(frame, gaze)
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
- Trace actual visible boundaries. Do not draw diagonal edges around rectangular content. Use a
  four-corner rectangle for rectangular text, images, panels, and popups. Include the full visible
  text block, not a small excerpt, and never cover a different neighboring element.
- Preserve the real shape of non-rectangular elements. Use 8-16 boundary points for circles,
  curved buttons, diagrams, and irregular or concave figures; never replace them with a bounding box.
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
    title = str(article.get("title") or "article title")
    heading_pattern = re.compile(r"(?m)^([A-Z][A-Z ]{2,}):\s*")
    matches = list(heading_pattern.finditer(full_text))
    choices: List[Tuple[str, str]] = [(title, full_text[:matches[0].start()] if matches else full_text)]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        choices.append((match.group(1).strip(), full_text[match.end():end]))

    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "it", "that", "this", "for", "with", "on", "be", "as"}
    def tokens(text: str) -> set[str]:
        return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in stop}
    choice_tokens = [(label, tokens(body)) for label, body in choices]
    assignments = {}
    for sample in samples:
        observed = " ".join(str(item.get("text") or "") for item in sample.ocr)
        observed_tokens = tokens(observed)
        scores = []
        for label, section_tokens in choice_tokens:
            overlap = len(observed_tokens & section_tokens)
            score = overlap / max(1.0, np.sqrt(len(section_tokens)))
            scores.append((score, label))
        assignments[sample.sample_id] = max(scores, default=(0.0, title))[1] if observed_tokens else title
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
                cv2.putText(canvas, _display_text(element["label"])[:50], (int(x), max(18, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
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


def _library_regions(regions: Sequence[Dict[str, Any]]) -> List[List[List[float]]]:
    polygons = []
    for region in regions:
        if region.get("shape") == "rectangle":
            x0, x1 = region["x_min"], region["x_max"]
            y0, y1 = region["y_min"], region["y_max"]
            polygons.append(_rect_polygon(float(x0), float(y0), float(x1), float(y1)))
        elif region.get("shape") == "polygon":
            polygon = _normalize_polygon(region.get("points"))
            if polygon:
                polygons.append(polygon)
    return polygons


def _read_slide_csv_library(standard_dir: Path) -> List[Dict[str, Any]]:
    manifest = standard_dir / "slides.csv"
    if not manifest.exists():
        raise RuntimeError(f"Prepared slide library not found: {manifest}")
    slides = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for slide in csv.DictReader(handle):
            decisions: Dict[str, Dict[str, Any]] = {}
            table = standard_dir / "elements" / slide["element_table"]
            with table.open(newline="", encoding="utf-8") as element_handle:
                for row in csv.DictReader(element_handle):
                    key = row["element_id"]
                    item = decisions.setdefault(key, {
                        "element_id": key, "element_type": row["element_type"],
                        "label": row["decision"], "priority": int(row["priority"]), "polygons": [],
                    })
                    if row["shape"] == "rectangle":
                        item["polygons"].append(_rect_polygon(
                            float(row["x_min"]), float(row["y_min"]),
                            float(row["x_max"]), float(row["y_max"]),
                        ))
                    elif row["shape"] == "ellipse":
                        item["polygons"].append(_ellipse_polygon(
                            float(row["center_x"]), float(row["center_y"]),
                            float(row["radius_x"]), float(row["radius_y"]),
                        ))
                    else:
                        points = [[float(v) for v in pair.split(":")] for pair in row["points"].split(";") if pair]
                        if len(points) >= 3:
                            item["polygons"].append(points)
            slides.append({
                "slide_id": slide["slide_id"], "description": slide["description"],
                "reference_images": (slide.get("reference_images") or slide.get("reference_image") or "").split(";"),
                "elements": list(decisions.values()),
            })
    return slides


def analyze_slides_with_standard_library(
    video_name: str, samples: Sequence[Sample], model: str, course_roi: Sequence[Sequence[float]],
    standard_dir: Path, match_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Select participant states from the frozen course-level decision library."""
    standard_dir.mkdir(parents=True, exist_ok=True)
    template_dir = standard_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    catalog = _read_slide_csv_library(standard_dir)
    templates = []
    for state in catalog:
        for filename in state["reference_images"]:
            image = cv2.imread(str(standard_dir / "references" / filename))
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
        details = ", ".join(f"{sample.pattern_id}@{sample.timestamp:.3f}s" for sample in unmatched)
        raise RuntimeError(
            "Slide decision must be selected from the frozen standard library; "
            f"no permitted match was found for: {details}. Update the library separately before rerunning."
        )

    states = []
    for pattern_id in by_pattern:
        standard = matched[pattern_id]
        states.append({
            "sample_id": pattern_id,
            "content_id": standard["slide_id"],
            "state_id": standard["slide_id"],
            "state_description": standard.get("description", ""),
            "elements": standard["elements"],
        })
    return {"video_type": "slides", "summary": "States matched to the shared course slide library.", "article": {}, "states": states}


def prepare_slide_standard_library(
    video_name: str, samples: Sequence[Sample], model: str,
    course_roi: Sequence[Sequence[float]], standard_dir: Path, merge_threshold: float = 3.0,
) -> None:
    """Pre-operation: merge duplicates, learn elements, and write a CSV-only library."""
    representatives: List[Sample] = []
    variants: Dict[str, List[Sample]] = {}
    for sample in samples:
        crop = _crop_to_roi(sample.clean_frame, course_roi)
        matches = [(_crop_difference(crop, _crop_to_roi(other.clean_frame, course_roi)), other) for other in representatives]
        best = min(matches, key=lambda item: item[0]) if matches else None
        if best is None or best[0] > merge_threshold:
            representatives.append(sample)
            variants[sample.pattern_id] = [sample]
        else:
            variants[best[1].pattern_id].append(sample)
    if len(representatives) < 22:
        raise RuntimeError(f"Only {len(representatives)} unique slide states found; at least 22 are required")

    learned_by_id: Dict[str, Dict[str, Any]] = {}
    # One screenshot per vision task avoids cross-slide coordinate confusion.
    def learn_one(sample: Sample) -> Tuple[str, Dict[str, Any]]:
        state: Dict[str, Any] = {}
        for _ in range(3):
            result = analyze_with_ai(video_name, [sample], model, "slides", 1, course_roi)
            state = next(iter(result.get("states", [])), {})
            if any(
                str(element.get("element_type")) != "blank_area"
                and any(_normalize_polygon(polygon) for polygon in element.get("polygons", []))
                for element in state.get("elements", [])
            ):
                return sample.pattern_id, state
        raise RuntimeError(f"AI returned no elements for {sample.pattern_id} after three attempts")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(learn_one, sample) for sample in representatives]
        for future in as_completed(futures):
            pattern_id, state = future.result()
            learned_by_id[pattern_id] = state
    references = standard_dir / "references"
    templates = standard_dir / "templates"
    elements_dir = standard_dir / "elements"
    for directory in (references, templates, elements_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.iterdir():
            if stale.is_file():
                stale.unlink()
    manifest_rows = []
    colors = {key: value for key, value in COLORS.items()}
    for number, sample in enumerate(representatives, 1):
        slide_id = f"slide_{number:03d}"
        raw = learned_by_id.get(sample.pattern_id, {})
        crop = _crop_to_roi(sample.clean_frame, course_roi)
        template_name, table_name = f"{slide_id}.png", f"{slide_id}.csv"
        reference_names = []
        for variant_number, variant in enumerate(variants[sample.pattern_id], 1):
            reference_name = f"{slide_id}_ref_{variant_number:02d}.png"
            cv2.imwrite(str(references / reference_name), _crop_to_roi(variant.clean_frame, course_roi))
            reference_names.append(reference_name)
        canvas = cv2.addWeighted(crop, 0.5, np.full_like(crop, 255), 0.5, 0)
        rows = []
        raw_elements = list(raw.get("elements") or [])
        if not any(str(e.get("element_type")) == "blank_area" for e in raw_elements):
            raw_elements.append({
                "element_id": f"{slide_id}_blank", "element_type": "blank_area",
                "label": "blank area", "priority": -100,
                "polygons": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
            })
        h, w = crop.shape[:2]
        for element_number, element in enumerate(raw_elements, 1):
            element_id = f"{slide_id}_element_{element_number:02d}"
            element_type = str(element.get("element_type") or "blank_area")
            label = str(element.get("label") or element_type)
            priority = int(element.get("priority") or 0)
            for polygon in element.get("polygons", []):
                polygon = _normalize_polygon(polygon)
                if not polygon:
                    continue
                xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
                corners = {
                    (min(xs), min(ys)), (max(xs), min(ys)),
                    (max(xs), max(ys)), (min(xs), max(ys)),
                }
                # Only a truly axis-aligned four-corner region is a rectangle.
                # Every other boundary is preserved as a polygon.
                is_rect = len(polygon) == 4 and {(x, y) for x, y in polygon} == corners
                is_ellipse = element_type == "button" and 0.5 <= (max(xs) - min(xs)) / max(1e-6, max(ys) - min(ys)) <= 2.0
                if is_ellipse:
                    polygon = _ellipse_polygon(
                        (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                        (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2,
                    )
                rows.append({
                    "element_id": element_id, "decision": label, "element_type": element_type,
                    "priority": priority, "shape": "ellipse" if is_ellipse else ("rectangle" if is_rect else "polygon"),
                    "x_min": round(min(xs), 4) if is_rect and not is_ellipse else "", "x_max": round(max(xs), 4) if is_rect and not is_ellipse else "",
                    "y_min": round(min(ys), 4) if is_rect and not is_ellipse else "", "y_max": round(max(ys), 4) if is_rect and not is_ellipse else "",
                    "center_x": round((min(xs) + max(xs)) / 2, 4) if is_ellipse else "",
                    "center_y": round((min(ys) + max(ys)) / 2, 4) if is_ellipse else "",
                    "radius_x": round((max(xs) - min(xs)) / 2, 4) if is_ellipse else "",
                    "radius_y": round((max(ys) - min(ys)) / 2, 4) if is_ellipse else "",
                    "points": "" if is_rect or is_ellipse else ";".join(f"{x:.4f}:{y:.4f}" for x, y in polygon),
                })
                if element_type != "blank_area":
                    pts = np.array([[round(x * w), round(y * h)] for x, y in polygon], np.int32)
                    color = colors.get(element_type, (0, 0, 0))
                    cv2.polylines(canvas, [pts], True, color, 7, cv2.LINE_AA)
                    tx, ty = pts[0]
                    cv2.putText(canvas, _display_text(label)[:45], (int(tx), max(25, int(ty) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
        fields = ["element_id", "decision", "element_type", "priority", "shape", "x_min", "x_max", "y_min", "y_max", "center_x", "center_y", "radius_x", "radius_y", "points"]
        with (elements_dir / table_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        cv2.imwrite(str(templates / template_name), canvas)
        manifest_rows.append({
            "slide_id": slide_id, "description": str(raw.get("state_description") or ""),
            "reference_images": ";".join(reference_names), "template_image": template_name, "element_table": table_name,
        })
    with (standard_dir / "slides.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["slide_id", "description", "reference_images", "template_image", "element_table"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(manifest_rows)
    print(f"Prepared {len(representatives)} merged slide states in {standard_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI video classification, semantic layout library, and report")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gaze-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--standard-library-dir", type=Path)
    parser.add_argument("--prepare-slide-library", action="store_true")
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
    if args.prepare_slide_library:
        if args.type != "slides" or not args.standard_library_dir:
            raise RuntimeError("--prepare-slide-library requires --type slides and --standard-library-dir")
        prepare_slide_standard_library(args.video.name, samples, args.model, course_roi, args.standard_library_dir)
        return
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
