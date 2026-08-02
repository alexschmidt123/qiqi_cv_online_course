#!/usr/bin/env python3
"""Step 4: render sampled layout, gaze, label, and decision validation images."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from step_3_map_gaze_to_events import Element, State, _element_at, _state_at, load_library
except ImportError:  # imported as script.step_4_validate_attention
    from script.step_3_map_gaze_to_events import Element, State, _element_at, _state_at, load_library

COLORS = {
    "web_navigation": (255, 80, 40), "web_panel": (255, 180, 40),
    "webpage_title": (30, 80, 255), "slide_title": (30, 80, 255),
    "paragraph": (40, 180, 40), "image": (220, 70, 220),
    "button": (0, 170, 255), "popup": (180, 40, 255),
    "slide_navigation": (255, 150, 0), "blank_area": (150, 150, 150),
    "outside_course": (90, 90, 90),
}


def _load_gaze(path: Path) -> List[Tuple[float, float, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((float(row["timestamp_sec"]), float(row["x_bl"]), float(row["y_bl"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _sample_rows(rows: Sequence[Tuple[float, float, float]], count: int) -> List[Tuple[float, float, float]]:
    if not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(i)] for i in indices]


def _one_row_per_slide(rows: Sequence[Tuple[float, float, float]], states: Sequence[State]) -> List[Tuple[float, float, float]]:
    selected, seen = [], set()
    for state in states:
        if state.state_id in seen:
            continue
        seen.add(state.state_id)
        candidates = [row for row in rows if state.start_sec <= row[0] < state.end_sec]
        if candidates:
            midpoint = (state.start_sec + state.end_sec) / 2
            selected.append(min(candidates, key=lambda row: abs(row[0] - midpoint)))
        else:
            selected.append((state.start_sec, float("nan"), float("nan")))
    return selected


def _draw_text_panel(frame: np.ndarray, lines: Sequence[str]) -> None:
    panel_h = 34 + 28 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line[:140], (18, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)


def _decision(state: Optional[State], element: Optional[Element], nx: float, ny: float) -> List[str]:
    if state is None:
        return ["Decision: none", "Reason: timestamp is outside analyzed article/slide states"]
    if element is None:
        return ["Decision: none", f"Reason: gaze ({nx:.3f}, {ny:.3f}) is outside all element polygons"]
    if element.element_type == "outside_course":
        return ["Decision: unrelated content", "Reason: gaze is outside the detected online-course content ROI"]
    return [
        f"Decision: {element.label} [{element.element_type}]",
        f"Reason: gaze ({nx:.3f}, {ny:.3f}) is inside its polygon; priority={element.priority}",
    ]


def render(video: Path, library_path: Path, gaze_path: Path, output_dir: Path, count: int) -> None:
    library_data = json.loads(library_path.read_text(encoding="utf-8"))
    video_type = str(library_data.get("video_type") or "slides")
    width, height, states = load_library(library_path)
    all_gaze_rows = _load_gaze(gaze_path)
    gaze_rows = _one_row_per_slide(all_gaze_rows, states) if video_type == "slides" else _sample_rows(all_gaze_rows, count)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (timestamp, x_bl, y_bl) in enumerate(gaze_rows, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        has_gaze = np.isfinite(x_bl) and np.isfinite(y_bl)
        x_tl, y_tl = (x_bl, (h - 1) - y_bl) if has_gaze else (0.0, 0.0)
        nx, ny = x_tl / width, y_tl / height
        state = _state_at(states, timestamp)
        element = _element_at(state, nx, ny) if state and has_gaze else None
        canvas = cv2.addWeighted(frame, 0.5, np.full_like(frame, 255), 0.5, 0)
        if state and video_type == "slides":
            for roi in state.course_roi:
                roi_points = np.array([[round(px * w), round(py * h)] for px, py in roi], np.int32)
                cv2.polylines(canvas, [roi_points], True, (255, 255, 0), 4, cv2.LINE_AA)
            for candidate in state.elements:
                color = COLORS.get(candidate.element_type, (0, 0, 0))
                thickness = 6 if element and candidate.element_id == element.element_id else 3
                for polygon in candidate.polygons:
                    points = np.array([[round(px * w), round(py * h)] for px, py in polygon], np.int32)
                    cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)
                    tx, ty = points[0]
                    cv2.putText(canvas, candidate.label[:45], (int(tx), max(90, int(ty) - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        if has_gaze:
            cv2.circle(canvas, (round(x_tl), round(y_tl)), 13, (0, 0, 255), -1, cv2.LINE_AA)
        state_name = state.state_id if state else "none"
        if video_type == "article" and element and element.element_type == "article_section":
            decision = [f"Decision: {element.label}", "Reason: gaze-near OCR words matched this section in full_article.txt (NLP)"]
        else:
            decision = _decision(state, element, nx, ny)
        _draw_text_panel(canvas, [f"Time: {timestamp:.3f}s | Visible state: {state_name}", *decision])
        suffix = state_name if video_type == "slides" else f"{timestamp:.3f}s"
        cv2.imwrite(str(output_dir / f"validation_{index:03d}_{suffix}.png"), canvas)
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render visual validation samples")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--gaze-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    render(args.video, args.library, args.gaze_data, args.output_dir, args.samples)


if __name__ == "__main__":
    main()
