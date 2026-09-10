#!/usr/bin/env python3
"""Offline: build shared slide reference library from all video_slide videos.

SOP: strip red gaze + cursor, cluster content states, ChatGPT labels each once,
freeze one reference + one element CSV per state. Online uses this library only.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

from step_2_analyze_video_with_ai import (  # noqa: E402
    COLORS,
    RECTANGULAR_ELEMENT_TYPES,
    _axis_aligned_rect_from_polygon,
    _crop_difference,
    _crop_to_roi,
    _display_text,
    _ellipse_polygon,
    _locate_content_roi,
    _normalize_polygon,
    _remove_overlay_markers,
    segment_one_slide_crop_with_sop,
)


def _sample_video_crops(
    video: Path,
    interval: float,
    change_threshold: float,
    max_patterns: int,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], List[List[float]]]:
    """Sample ROI crops with gaze/cursor stripped; keep unique content patterns."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_interval = max(1, int(round(interval * fps)))
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError(f"No frames in {video}")
    course_roi = _locate_content_roi(first, "slides")

    crops: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    reps: List[np.ndarray] = []
    frame_idx = 0
    # Re-process first frame as sample 0.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            clean = _remove_overlay_markers(frame, gaze=None)
            crop = _crop_to_roi(clean, course_roi)
            if crop.size == 0:
                frame_idx += 1
                continue
            if reps:
                best = min(_crop_difference(crop, r) for r in reps)
                if best < change_threshold:
                    frame_idx += 1
                    continue
            if len(reps) >= max_patterns:
                break
            reps.append(crop)
            crops.append(crop)
            metas.append({"video": video.name, "timestamp": float(timestamp), "index": len(crops)})
            print(f"    + pattern {len(crops)} @ {timestamp:.1f}s", flush=True)
        frame_idx += 1
    cap.release()
    return crops, metas, course_roi


def _cluster_crops(
    crops: Sequence[np.ndarray],
    metas: Sequence[Dict[str, Any]],
    merge_threshold: float,
) -> List[Dict[str, Any]]:
    reps: List[Dict[str, Any]] = []
    for crop, meta in zip(crops, metas):
        best_i, best_d = -1, 1e9
        for i, rep in enumerate(reps):
            d = _crop_difference(crop, rep["crop"])
            if d < best_d:
                best_i, best_d = i, d
        if best_i < 0 or best_d > merge_threshold:
            reps.append({"crop": crop, "meta": meta, "members": 1})
        else:
            reps[best_i]["members"] += 1
    return reps


def _write_state(
    standard_dir: Path,
    state_id: str,
    description: str,
    crop: np.ndarray,
    elements: List[Dict[str, Any]],
) -> Dict[str, str]:
    references = standard_dir / "references"
    templates = standard_dir / "templates"
    elements_dir = standard_dir / "elements"
    for d in (references, templates, elements_dir):
        d.mkdir(parents=True, exist_ok=True)

    ref_name = f"{state_id}_ref_01.png"
    tmpl_name = f"{state_id}.png"
    table_name = f"{state_id}.csv"
    cv2.imwrite(str(references / ref_name), crop)

    canvas = cv2.addWeighted(crop, 0.5, np.full_like(crop, 255), 0.5, 0)
    h, w = crop.shape[:2]
    rows = []
    raw = list(elements)
    if not any(str(e.get("element_type")) == "blank_area" for e in raw):
        raw.append(
            {
                "element_type": "blank_area",
                "label": "blank area",
                "priority": -100,
                "polygons": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
            }
        )
    for n, element in enumerate(raw, 1):
        element_type = str(element.get("element_type") or "blank_area")
        label = str(element.get("label") or element_type)
        priority = int(element.get("priority") or 0)
        for polygon in element.get("polygons") or []:
            polygon = _normalize_polygon(polygon)
            if not polygon:
                continue
            xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
            force_rect = element_type in RECTANGULAR_ELEMENT_TYPES
            is_ellipse = (
                element_type == "button"
                and not force_rect
                and 0.5 <= (max(xs) - min(xs)) / max(1e-6, max(ys) - min(ys)) <= 2.0
            )
            if force_rect:
                polygon = _axis_aligned_rect_from_polygon(polygon)
                xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
                is_rect = True
            else:
                corners = {
                    (min(xs), min(ys)),
                    (max(xs), min(ys)),
                    (max(xs), max(ys)),
                    (min(xs), max(ys)),
                }
                is_rect = len(polygon) == 4 and {(x, y) for x, y in polygon} == corners
            if is_ellipse:
                polygon = _ellipse_polygon(
                    (min(xs) + max(xs)) / 2,
                    (min(ys) + max(ys)) / 2,
                    (max(xs) - min(xs)) / 2,
                    (max(ys) - min(ys)) / 2,
                )
            rows.append(
                {
                    "element_id": f"{state_id}_element_{n:02d}",
                    "decision": label,
                    "element_type": element_type,
                    "priority": priority,
                    "shape": "ellipse" if is_ellipse else ("rectangle" if is_rect else "polygon"),
                    "x_min": round(min(xs), 4) if is_rect and not is_ellipse else "",
                    "x_max": round(max(xs), 4) if is_rect and not is_ellipse else "",
                    "y_min": round(min(ys), 4) if is_rect and not is_ellipse else "",
                    "y_max": round(max(ys), 4) if is_rect and not is_ellipse else "",
                    "center_x": round((min(xs) + max(xs)) / 2, 4) if is_ellipse else "",
                    "center_y": round((min(ys) + max(ys)) / 2, 4) if is_ellipse else "",
                    "radius_x": round((max(xs) - min(xs)) / 2, 4) if is_ellipse else "",
                    "radius_y": round((max(ys) - min(ys)) / 2, 4) if is_ellipse else "",
                    "points": ""
                    if is_rect or is_ellipse
                    else ";".join(f"{x:.4f}:{y:.4f}" for x, y in polygon),
                }
            )
            if element_type != "blank_area":
                pts = np.array([[round(x * w), round(y * h)] for x, y in polygon], np.int32)
                color = COLORS.get(element_type, (0, 0, 0))
                cv2.polylines(canvas, [pts], True, color, 5, cv2.LINE_AA)
                tx, ty = pts[0]
                cv2.putText(
                    canvas,
                    _display_text(label)[:40],
                    (int(tx), max(22, int(ty) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    fields = [
        "element_id",
        "decision",
        "element_type",
        "priority",
        "shape",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "center_x",
        "center_y",
        "radius_x",
        "radius_y",
        "points",
    ]
    with (elements_dir / table_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(templates / tmpl_name), canvas)
    return {
        "slide_id": state_id,
        "description": description,
        "reference_images": ref_name,
        "template_image": tmpl_name,
        "element_table": table_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline multi-video slide library build")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video_slide")
    parser.add_argument(
        "--standard-library-dir",
        type=Path,
        default=ROOT / "data" / "slide_standard_library",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.7,
        help="Per-video uniqueness threshold (keep popups).",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=1.2,
        help="Cross-video merge after gaze/cursor strip.",
    )
    parser.add_argument("--max-patterns-per-video", type=int, default=80)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cluster-only", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument(
        "--from-preview",
        type=Path,
        help="Skip video sampling; cluster/label PNGs from this preview directory.",
    )
    args = parser.parse_args()

    if args.from_preview:
        preview = args.from_preview
        paths = sorted(preview.glob("slide_*.png")) + sorted(preview.glob("state_*.png"))
        if not paths:
            raise SystemExit(f"No preview PNGs in {preview}")
        all_crops, all_metas = [], []
        for p in paths:
            im = cv2.imread(str(p))
            if im is None:
                continue
            all_crops.append(im)
            all_metas.append({"video": "preview", "timestamp": 0.0, "index": len(all_crops), "name": p.name})
        print(f"[offline] loaded {len(all_crops)} preview crops from {preview}")
    else:
        videos = sorted(list(args.video_dir.glob("*.mp4")) + list(args.video_dir.glob("*.mov")))
        if not videos:
            raise SystemExit(f"No videos in {args.video_dir}")

        all_crops = []
        all_metas = []
        for video in videos:
            print(f"[offline] sampling {video.name} …")
            crops, metas, _roi = _sample_video_crops(
                video,
                args.sample_interval,
                args.change_threshold,
                args.max_patterns_per_video,
            )
            print(f"  unique patterns: {len(crops)}")
            all_crops.extend(crops)
            all_metas.extend(metas)

    reps = _cluster_crops(all_crops, all_metas, args.merge_threshold)
    print(f"[offline] unique content states across all videos: {len(reps)}")
    for i, rep in enumerate(reps, 1):
        m = rep["meta"]
        print(
            f"  slide_{i:03d}: members≈{rep['members']} "
            f"from {m['video']} @{m['timestamp']:.1f}s"
        )

    if args.cluster_only:
        preview = args.standard_library_dir.parent / "slide_library_cluster_preview"
        if preview.exists():
            shutil.rmtree(preview)
        preview.mkdir(parents=True)
        for i, rep in enumerate(reps, 1):
            cv2.imwrite(str(preview / f"slide_{i:03d}.png"), rep["crop"])
        print(f"[offline] wrote cluster preview to {preview}")
        return

    lib = args.standard_library_dir
    if not args.no_backup and lib.exists() and any(lib.iterdir()):
        backup = lib.parent / "slide_standard_library_backup"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(lib, backup)
        print(f"[offline] backed up previous library → {backup}")

    for sub in ("references", "templates", "elements"):
        d = lib / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    def learn(rep: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        crop = rep["crop"]
        state: Dict[str, Any] = {}
        for _ in range(3):
            state = segment_one_slide_crop_with_sop(crop, args.model)
            if any(
                str(e.get("element_type")) != "blank_area"
                and any(_normalize_polygon(p) for p in e.get("polygons") or [])
                for e in state.get("elements") or []
            ):
                return rep, state
        raise RuntimeError(f"AI returned no elements for {rep['meta']}")

    print(f"[offline] labeling {len(reps)} states with {args.model} …")
    learned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(learn, rep) for rep in reps]
        for fut in as_completed(futures):
            learned.append(fut.result())
            print(f"  labeled {len(learned)}/{len(reps)}")

    order = {id(rep["crop"]): i for i, rep in enumerate(reps)}
    learned.sort(key=lambda item: order[id(item[0]["crop"])])

    manifest = []
    for i, (rep, state) in enumerate(learned, 1):
        state_id = f"slide_{i:03d}"
        manifest.append(
            _write_state(
                lib,
                state_id,
                str(state.get("state_description") or ""),
                rep["crop"],
                list(state.get("elements") or []),
            )
        )
        print(f"  wrote {state_id}")

    with (lib / "slides.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["slide_id", "description", "reference_images", "template_image", "element_table"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"[offline] done: {len(manifest)} states → {lib}")


if __name__ == "__main__":
    main()
