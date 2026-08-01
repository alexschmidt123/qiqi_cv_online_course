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

DEFAULT_MODEL = "gpt-5.6-sol"
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
    timestamp: float
    frame: np.ndarray
    clean_frame: np.ndarray
    ocr: List[Dict[str, Any]]


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


def _visual_difference(a: np.ndarray, b: np.ndarray) -> float:
    def prep(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(prep(a), prep(b))))


def _ocr_frame(reader: Any, frame: np.ndarray, width: int, height: int) -> List[Dict[str, Any]]:
    if reader is None:
        return []
    rows = []
    for bbox, text, confidence in reader.readtext(frame):
        if not str(text).strip() or float(confidence or 0) < 0.08:
            continue
        points = [[round(float(x) / width, 5), round(float(y) / height, 5)] for x, y in bbox]
        rows.append({"text": str(text).strip(), "confidence": round(float(confidence), 3), "polygon": points})
    return rows


def collect_samples(video: Path, interval: float, change_threshold: float, max_samples: int) -> Tuple[int, int, float, List[Sample]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps if count else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader = easyocr.Reader(["en"], gpu=False, verbose=False) if easyocr is not None else None
    samples: List[Sample] = []
    previous: Optional[np.ndarray] = None
    timestamps = np.arange(0.0, max(duration, interval), interval).tolist()
    if len(timestamps) > max_samples:
        timestamps = np.linspace(0.0, max(0.0, duration - 1.0 / fps), max_samples).tolist()
    for index, timestamp in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        clean = _remove_red_gaze(frame)
        # Keep all scrolling samples; collapse nearly identical slide states.
        diff = _visual_difference(previous, clean) if previous is not None else 999.0
        if previous is not None and diff < change_threshold:
            continue
        sample_id = f"sample_{len(samples) + 1:03d}"
        samples.append(Sample(sample_id, float(timestamp), frame, clean, _ocr_frame(reader, clean, width, height)))
        previous = clean
    cap.release()
    if not samples:
        raise RuntimeError("No representative frames could be read")
    return width, height, duration, samples


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


def analyze_with_ai(video_name: str, samples: Sequence[Sample], model: str, requested_type: str) -> Dict[str, Any]:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for step 2 layout analysis")
    evidence = {s.sample_id: {"timestamp_sec": s.timestamp, "ocr": s.ocr} for s in samples}
    prompt = f"""
Analyze representative screenshots from the online-course video {video_name!r}.
Requested video type: {requested_type}. If it is auto, classify it as exactly "article" or "slides".

OCR evidence (normalized top-left coordinates) follows. Correct OCR mistakes using the images:
{json.dumps(evidence, ensure_ascii=False)}

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
      "slide_id": "stable logical slide/page id",
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
- Every polygon uses normalized top-left coordinates from 0 to 1 and may be concave.
- Use multiple polygons for disconnected regions.
- Include fixed website navigation/panels as elements.
- For article videos, reconstruct the complete article across scrolling screenshots, remove duplicates,
  preserve reading order, and label title/headings/paragraphs/images.
- For slides, include every distinct slide state. A popup revealed by a clickable button is a new
  state even when the base slide is unchanged. Give popup elements higher priority than covered content.
- Do not treat the red gaze marker as content.
- Produce one state for each supplied sample_id. Reuse slide_id when samples show the same base slide.
"""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for sample in samples:
        content.append({"type": "input_text", "text": f"{sample.sample_id} at {sample.timestamp:.3f}s"})
        content.append({"type": "input_image", "image_url": _jpeg_data_url(sample.clean_frame), "detail": "high"})
    client = OpenAI()
    response = client.responses.create(model=model, input=[{"role": "user", "content": content}])
    return _extract_json(response.output_text)


def _normalize_polygon(raw: Any) -> List[List[float]]:
    points = []
    if not isinstance(raw, list):
        return points
    for point in raw:
        if isinstance(point, list) and len(point) >= 2:
            points.append([max(0.0, min(1.0, float(point[0]))), max(0.0, min(1.0, float(point[1])))])
    return points if len(points) >= 3 else []


def _point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x <= (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _text_roi_from_ocr(ai_polygons: Sequence[Sequence[Sequence[float]]], ocr: Sequence[Dict[str, Any]]) -> List[List[List[float]]]:
    """Use AI for semantics/layout, but OCR+CV boxes for the actual text hit regions."""
    hits: List[List[List[float]]] = []
    for item in ocr:
        polygon = _normalize_polygon(item.get("polygon"))
        if not polygon:
            continue
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        if any(_point_in_polygon(cx, cy, region) for region in ai_polygons):
            # Add a small gaze tolerance around OCR's quadrilateral.
            xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
            pad_x, pad_y = 0.006, 0.008
            hits.append([
                [max(0.0, min(xs) - pad_x), max(0.0, min(ys) - pad_y)],
                [min(1.0, max(xs) + pad_x), max(0.0, min(ys) - pad_y)],
                [min(1.0, max(xs) + pad_x), min(1.0, max(ys) + pad_y)],
                [max(0.0, min(xs) - pad_x), min(1.0, max(ys) + pad_y)],
            ])
    return hits


def build_library(result: Dict[str, Any], samples: Sequence[Sample], width: int, height: int, duration: float) -> Dict[str, Any]:
    by_id = {str(s.get("sample_id")): s for s in result.get("states", []) if isinstance(s, dict)}
    states = []
    for index, sample in enumerate(samples):
        raw = by_id.get(sample.sample_id, {})
        end = samples[index + 1].timestamp if index + 1 < len(samples) else duration + 1e-6
        elements = []
        for number, element in enumerate(raw.get("elements", []), 1):
            polygons = [_normalize_polygon(p) for p in element.get("polygons", [])]
            polygons = [p for p in polygons if p]
            if not polygons:
                continue
            element_type = str(element.get("element_type") or "blank_area")
            if element_type in {"web_navigation", "webpage_title", "slide_title", "paragraph", "button"}:
                ocr_polygons = _text_roi_from_ocr(polygons, sample.ocr)
                if ocr_polygons:
                    polygons = ocr_polygons
            elements.append({
                "element_id": str(element.get("element_id") or f"{sample.sample_id}_element_{number:02d}"),
                "element_type": element_type,
                "label": str(element.get("label") or element.get("element_type") or "unknown"),
                "priority": int(element.get("priority") or 0),
                "polygons": polygons,
            })
        states.append({
            "sample_id": sample.sample_id,
            "slide_id": str(raw.get("slide_id") or sample.sample_id),
            "state_id": str(raw.get("state_id") or sample.sample_id),
            "state_description": str(raw.get("state_description") or ""),
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
    for state in library["states"]:
        sample = sample_by_id.get(state["sample_id"])
        if sample is None:
            continue
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
        lines += ["## Article", "", f"Title: {article.get('title', '')}", "", "### Full text", "", str(article.get("full_text", "")), "", "### Layout labels", ""]
        for item in article.get("layout_labels", []):
            lines.append(f"- {item.get('label', '')}: {item.get('text', '')} ({item.get('sample_id', '')})")
    else:
        lines += ["## Slide states", ""]
        for state in library["states"]:
            lines.append(f"- {state['state_id']} — {state['state_description']} (`layout/{state['state_id']}.png`)")
    (output_dir / "ai_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI video classification, semantic layout library, and report")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--type", choices=("auto", "article", "slides"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--change-threshold", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, height, duration, samples = collect_samples(args.video, args.sample_interval, args.change_threshold, args.max_samples)
    result = analyze_with_ai(args.video.name, samples, args.model, args.type)
    if args.type != "auto":
        result["video_type"] = args.type
    library = build_library(result, samples, width, height, duration)
    (args.output_dir / "element_library.json").write_text(json.dumps(library, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_layout_images(samples, library, args.output_dir / "layout")
    write_report(result, library, args.output_dir)
    print(f"Wrote {args.output_dir / 'element_library.json'} and AI report")


if __name__ == "__main__":
    main()
