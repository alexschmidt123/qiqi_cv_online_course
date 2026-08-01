"""Legacy article validation renderer retained for reproducibility.
Step 4 (optional): Visual validation of final gaze + section per video.

For one video:
- Input:
    - video file
    - final_gaze_table.csv (from legacy step_3_refine_with_gpt.py)
    - optional debug/step2_bucket_debug.csv (same output folder as the pipeline) for scrollable-area box
- Output (in output/<video_name>/validation/):
    - frame_1s.png
    - frame_mid.png
    - frame_end.png

For each target time (≈1s, mid, end-1s):
- Find the nearest frame at or after that time that has gaze.
- Draw:
    - light blue rectangle = scrollable article region (from step 2 OCR estimate for that time bucket;
      dimensions differ per video and can change between buckets)
    - yellow marker at the detected gaze coordinate
    - text: section = <section_name> from final_gaze_table.csv
  (the underlying red circle from the video acts as ground truth for visual comparison).
"""
import argparse
import csv
import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Light blue border (BGR) for scrollable article box — drawn under gaze overlay.
SCROLLABLE_BORDER_BGR = (230, 216, 173)  # ~ #ADD8E6 light blue
SCROLLABLE_BORDER_THICKNESS = 2

DEFAULT_INTERVAL_SEC = 1.0


def _load_final_table(path: str) -> List[Tuple[int, float, Optional[float], Optional[float], str]]:
    rows: List[Tuple[int, float, Optional[float], Optional[float], str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row.get("frame_idx", "0"))
            except ValueError:
                continue
            try:
                ts = float(row.get("timestamp_sec", "0.0"))
            except ValueError:
                ts = 0.0
            has_gaze = (row.get("has_gaze") or "").strip()
            x_s = (row.get("gaze_x") or "").strip()
            y_s = (row.get("gaze_y") or "").strip()
            sec = (row.get("section_name") or "").strip()
            if has_gaze and has_gaze != "0" and x_s and y_s:
                try:
                    x = float(x_s)
                    y = float(y_s)
                except ValueError:
                    x = y = None
            else:
                x = y = None
            rows.append((frame_idx, ts, x, y, sec))
    return rows


def _default_step2_debug_csv(validation_dir: str) -> str:
    """output/<name>/validation/ -> output/<name>/debug/step2_bucket_debug.csv"""
    parent = os.path.dirname(os.path.abspath(validation_dir))
    return os.path.join(parent, "debug", "step2_bucket_debug.csv")


def _load_article_boxes_by_bucket(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Load per-bucket scrollable article box in top-left pixel coords from step2_bucket_debug.csv.
    Returns bucket -> (x0_tl, y0_tl, x1_tl, y1_tl).
    """
    boxes: Dict[int, Tuple[float, float, float, float]] = {}
    if not os.path.isfile(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                b = int(row.get("bucket", ""))
            except (TypeError, ValueError):
                continue
            try:
                x0 = float(row["article_x0_tl"])
                y0 = float(row["article_y0_tl"])
                x1 = float(row["article_x1_tl"])
                y1 = float(row["article_y1_tl"])
            except (KeyError, TypeError, ValueError):
                continue
            boxes[b] = (x0, y0, x1, y1)
    return boxes


def _bucket_for_timestamp(ts: float, interval_sec: float) -> int:
    if interval_sec <= 0:
        return 0
    return int(ts / interval_sec)


def _draw_scrollable_border(
    frame: np.ndarray,
    x0_tl: float,
    y0_tl: float,
    x1_tl: float,
    y1_tl: float,
    color_bgr: Tuple[int, int, int] = SCROLLABLE_BORDER_BGR,
    thickness: int = SCROLLABLE_BORDER_THICKNESS,
) -> None:
    """Draw article box; coordinates are top-left origin (same as OpenCV image)."""
    h, w = frame.shape[:2]
    x0 = int(round(max(0, min(x0_tl, x1_tl))))
    y0 = int(round(max(0, min(y0_tl, y1_tl))))
    x1 = int(round(min(w - 1, max(x0_tl, x1_tl))))
    y1 = int(round(min(h - 1, max(y0_tl, y1_tl))))
    if x1 <= x0 or y1 <= y0:
        return
    cv2.rectangle(frame, (x0, y0), (x1, y1), color_bgr, thickness, lineType=cv2.LINE_AA)


def _find_frame_with_gaze(
    rows: List[Tuple[int, float, Optional[float], Optional[float], str]],
    target_time: float,
    max_delta_sec: float = 1.0,
) -> Optional[Tuple[int, float, float, float, str]]:
    """
    Find the first row at or after target_time that has gaze within max_delta_sec.
    Returns (frame_idx, ts, x_bl, y_bl, section) or None.
    """
    candidate = None
    for frame_idx, ts, x, y, sec in rows:
        if x is None or y is None:
            continue
        if ts < target_time:
            continue
        if ts - target_time > max_delta_sec:
            break
        candidate = (frame_idx, ts, x, y, sec)
        break
    return candidate


def _bl_to_image(x_bl: float, y_bl: float, height: int) -> Tuple[int, int]:
    x = int(round(x_bl))
    y_img = (height - 1) - y_bl
    return x, int(round(y_img))


def _draw_marker(
    frame: np.ndarray,
    x_bl: float,
    y_bl: float,
    section: str,
    color_bgr=(0, 255, 255),
) -> None:
    h, w = frame.shape[:2]
    x, y = _bl_to_image(x_bl, y_bl, h)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    cv2.circle(frame, (x, y), 6, color_bgr, -1)
    label = f"section = {section or '(none)'}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 4
    x1 = max(0, x + 8)
    y1 = max(0, y - th - pad)
    x2 = min(w, x1 + tw + pad * 2)
    y2 = min(h, y1 + th + pad * 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 1)
    cv2.putText(frame, label, (x1 + pad, y2 - pad), font, scale, color_bgr, thickness, cv2.LINE_AA)


def run_validation(
    video_path: str,
    final_csv: str,
    validation_dir: str,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    step2_debug_csv: Optional[str] = None,
) -> None:
    rows = _load_final_table(final_csv)
    if not rows:
        print("No rows in final_gaze_table:", final_csv)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if frame_count > 0 else rows[-1][1]

    targets = [
        (1.0, "frame_1s.png"),
        (duration_sec / 2.0, "frame_mid.png"),
        (max(duration_sec - 1.0, 1.0), "frame_end.png"),
    ]

    dbg_path = step2_debug_csv or _default_step2_debug_csv(validation_dir)
    article_boxes = _load_article_boxes_by_bucket(dbg_path)
    if not article_boxes:
        print(
            "No scrollable boxes loaded (missing or empty step2 debug). "
            f"Expected: {dbg_path} — run step 2 first or pass --step2-debug-csv."
        )

    os.makedirs(validation_dir, exist_ok=True)

    for t_target, name in targets:
        chosen = _find_frame_with_gaze(rows, t_target, max_delta_sec=1.0)
        if chosen is None:
            print(f"No gaze found near t≈{t_target:.2f}s; skipping {name}")
            continue
        frame_idx, ts, x_bl, y_bl, sec = chosen
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx} for {name}")
            continue
        # Save original frame (no overlay)
        base, ext = os.path.splitext(name)
        original_name = base + "_original" + ext
        original_path = os.path.join(validation_dir, original_name)
        cv2.imwrite(original_path, frame.copy())
        # Scrollable article region for this time bucket (per-video geometry from step 2)
        b = _bucket_for_timestamp(ts, interval_sec)
        box = article_boxes.get(b)
        if box is not None:
            _draw_scrollable_border(frame, box[0], box[1], box[2], box[3])
        # Gaze + section label on top
        _draw_marker(frame, x_bl, y_bl, sec)
        out_path = os.path.join(validation_dir, name)
        cv2.imwrite(out_path, frame)
        print(f"Saved {original_name}, {name} at t={ts:.2f}s (frame {frame_idx}, section='{sec}')")

    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4 (optional): visual validation of final_gaze_table.csv (3 key frames)."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to video file (.mp4/.mov).")
    parser.add_argument(
        "--final-csv",
        type=str,
        required=True,
        help="Path to final_gaze_table.csv (from step_3_refine_with_gpt.py).",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        required=True,
        help="Folder to write validation images (e.g. output/VIDEO_NAME/validation).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        metavar="SEC",
        help="Same bucket length as step 2/3 (default: 1.0). Must match run.sh -interval.",
    )
    parser.add_argument(
        "--step2-debug-csv",
        type=str,
        default="",
        help="Path to debug/step2_bucket_debug.csv (default: <parent of validation-dir>/debug/step2_bucket_debug.csv).",
    )
    args = parser.parse_args()

    step2_arg = args.step2_debug_csv.strip() or None
    run_validation(
        os.path.abspath(args.video),
        os.path.abspath(args.final_csv),
        os.path.abspath(args.validation_dir),
        interval_sec=args.interval,
        step2_debug_csv=os.path.abspath(step2_arg) if step2_arg else None,
    )


if __name__ == "__main__":
    main()
