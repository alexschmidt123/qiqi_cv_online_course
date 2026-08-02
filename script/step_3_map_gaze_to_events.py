#!/usr/bin/env python3
"""Step 3: map gaze coordinates to the AI-built element library and emit dwell events.

The expensive/semantic work happens once when element_library.json is created.
This script is deterministic and can be reused for every participant recording.
Coordinates in the library are normalized top-left coordinates (0..1).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Element:
    element_id: str
    element_type: str
    label: str
    polygons: Tuple[Tuple[Point, ...], ...]
    priority: int = 0


@dataclass(frozen=True)
class State:
    state_id: str
    content_id: str
    start_sec: float
    end_sec: float
    course_roi: Tuple[Tuple[Point, ...], ...]
    elements: Tuple[Element, ...]


def _point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray casting test; concave polygons are supported."""
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        on_edge = abs((y - yi) * (xj - xi) - (x - xi) * (yj - yi)) < 1e-10
        if on_edge and min(xi, xj) <= x <= max(xi, xj) and min(yi, yj) <= y <= max(yi, yj):
            return True
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_at_y:
                inside = not inside
        j = i
    return inside


def _polygon_area(poly: Sequence[Point]) -> float:
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly))) / 2.0)


def load_library(path: Path) -> Tuple[int, int, List[State]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    width = int(data["frame_size"]["width"])
    height = int(data["frame_size"]["height"])
    states: List[State] = []
    for raw_state in data["states"]:
        elements = []
        for raw in raw_state.get("elements", []):
            polygons = tuple(tuple((float(x), float(y)) for x, y in p) for p in raw["polygons"])
            elements.append(Element(
                element_id=str(raw["element_id"]),
                element_type=str(raw["element_type"]),
                label=str(raw.get("label") or raw["element_id"]),
                polygons=polygons,
                priority=int(raw.get("priority", 0)),
            ))
        states.append(State(
            state_id=str(raw_state["state_id"]),
            content_id=str(raw_state.get("content_id") or raw_state.get("slide_id") or "unknown"),
            start_sec=float(raw_state["start_sec"]),
            end_sec=float(raw_state["end_sec"]),
            course_roi=tuple(
                tuple((float(x), float(y)) for x, y in polygon)
                for polygon in raw_state.get("course_roi", [[[0.22, 0.17], [0.78, 0.17], [0.78, 0.73], [0.22, 0.73]]])
            ),
            elements=tuple(elements),
        ))
    states.sort(key=lambda s: s.start_sec)
    return width, height, states


def _state_at(states: Sequence[State], timestamp: float) -> Optional[State]:
    # State intervals are [start, end), except the final endpoint is harmless in video data.
    return next((s for s in states if s.start_sec <= timestamp < s.end_sec), None)


def _element_at(state: State, nx: float, ny: float) -> Optional[Element]:
    if state.course_roi and not any(_point_in_polygon(nx, ny, polygon) for polygon in state.course_roi):
        return Element("unrelated_content", "outside_course", "unrelated content", tuple(), -1000)
    hits = []
    for element in state.elements:
        hit_areas = [_polygon_area(p) for p in element.polygons if _point_in_polygon(nx, ny, p)]
        if hit_areas:
            # Higher priority wins overlapping layers; smaller region breaks ties.
            hits.append((-element.priority, min(hit_areas), element.element_id, element))
    return min(hits)[-1] if hits else None


def _float(row: Dict[str, str], name: str) -> Optional[float]:
    value = (row.get(name) or "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def label_gaze_rows(gaze_csv: Path, width: int, height: int, states: Sequence[State], origin: str) -> List[Dict[str, Any]]:
    labeled: List[Dict[str, Any]] = []
    with gaze_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ts = _float(row, "timestamp_sec")
            x = _float(row, "x_bl")
            y = _float(row, "y_bl")
            if x is None: x = _float(row, "gaze_x")
            if y is None: y = _float(row, "gaze_y")
            state = _state_at(states, ts) if ts is not None else None
            element = None
            if state and x is not None and y is not None:
                y_tl = (height - 1) - y if origin == "bottom-left" else y
                element = _element_at(state, x / width, y_tl / height)
            labeled.append({
                "timestamp_sec": ts,
                "gaze_x": x,
                "gaze_y": y,
                "gaze_location": f"({x:.3f}, {y:.3f})" if x is not None and y is not None else "none",
                "content_id": state.content_id if state else "none",
                "state_id": state.state_id if state else "none",
                "element_id": element.element_id if element else "none",
                "element_type": element.element_type if element else "none",
                "learning_element": element.label if element else "none",
            })
    return labeled


def build_events(rows: Sequence[Dict[str, Any]], user_id: str, min_duration: float = 0.0) -> List[Dict[str, Any]]:
    """Run-length encode frame labels. Duration ends at the next sample timestamp."""
    valid = [r for r in rows if r["timestamp_sec"] is not None]
    if not valid:
        return []
    deltas = [valid[i + 1]["timestamp_sec"] - valid[i]["timestamp_sec"] for i in range(len(valid) - 1)]
    positive = sorted(d for d in deltas if d > 0)
    sample_period = positive[len(positive) // 2] if positive else 0.0
    events: List[Dict[str, Any]] = []
    start = 0
    keys = ("content_id", "state_id", "element_id")
    for i in range(1, len(valid) + 1):
        boundary = i == len(valid) or any(valid[i][k] != valid[start][k] for k in keys)
        if not boundary:
            continue
        first, last = valid[start], valid[i - 1]
        duration = max(0.0, last["timestamp_sec"] + sample_period - first["timestamp_sec"])
        if duration >= min_duration:
            events.append({
                "user_id": user_id,
                "content_id": first["content_id"],
                "state_id": first["state_id"],
                "time_point": round(first["timestamp_sec"], 6),
                "gaze_location": first["gaze_location"],
                "learning_element": first["learning_element"],
                "element_type": first["element_type"],
                "duration": round(duration, 6),
            })
        start = i
    return events


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], video_type: str) -> None:
    rows = list(rows)
    if video_type == "article":
        fields = ["time_point", "gaze_location", "section", "duration"]
    else:
        fields = ["time_point", "gaze_location", "slide_id", "element", "duration"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if video_type == "article":
                writer.writerow({
                    "time_point": row["time_point"], "gaze_location": row["gaze_location"],
                    "section": row["learning_element"],
                    "duration": row["duration"],
                })
            else:
                writer.writerow({
                    "time_point": row["time_point"], "gaze_location": row["gaze_location"],
                    # The visible state is the exported slide ID. Popup variants therefore
                    # receive distinct IDs such as slide_01_popup_button_1.
                    "slide_id": row["state_id"],
                    "element": row["learning_element"], "duration": row["duration"],
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Map gaze to learned element polygons and aggregate dwell events.")
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--gaze-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--origin", choices=("bottom-left", "top-left"), default="bottom-left")
    parser.add_argument("--min-duration", type=float, default=0.0)
    args = parser.parse_args()
    library_data = json.loads(args.library.read_text(encoding="utf-8"))
    video_type = str(library_data.get("video_type") or "slides")
    width, height, states = load_library(args.library)
    labeled = label_gaze_rows(args.gaze_csv, width, height, states, args.origin)
    write_csv(args.output, build_events(labeled, args.user_id, args.min_duration), video_type)


if __name__ == "__main__":
    main()
