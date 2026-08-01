"""
Step 1 (PPT / image videos): Red circle gaze per frame, bottom-left CSV.

Same detector as script/step_1_detect_gaze.py, but PPT videos have no REPLAY line:
only the bottom red navigation strip is excluded via OPERATION_BAR_TOP_MARGIN_PX.

Usage:
  python scripts_image/step_1_detect_gaze.py --video path.mp4 --output-dir output_image/NAME
"""
import argparse
import csv
import os
import sys

import cv2

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from script.step_1_detect_gaze import (  # noqa: E402
    OPERATION_BAR_TOP_MARGIN_PX,
    detect_red_gaze_point,
    to_bottom_left,
)


def process_video_ppt(video_path: str, output_csv_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    n_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames_est > 0:
        print(
            f"  PPT gaze: scanning ~{n_frames_est} frames @ {fps:.1f} fps (no output until done or every 500 frames)...",
            flush=True,
        )
    else:
        print("  PPT gaze: scanning frames (length unknown)...", flush=True)
    content_min_y_bl = float(OPERATION_BAR_TOP_MARGIN_PX)
    frame_idx = 0
    rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        ts = frame_idx / fps
        point, _n = detect_red_gaze_point(frame)
        if point is not None:
            x_tl, y_tl = point
            x_bl, y_bl = to_bottom_left(x_tl, y_tl, h)
            if y_bl < content_min_y_bl:
                rows.append((frame_idx, ts, "", ""))
            else:
                rows.append((frame_idx, ts, x_bl, y_bl))
        else:
            rows.append((frame_idx, ts, "", ""))
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  PPT gaze: processed {frame_idx} frames...", flush=True)
    cap.release()

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_sec", "x_bl", "y_bl"])
        for r in rows:
            w.writerow(list(r))
    return {"csv_path": output_csv_path, "total_frames": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PPT video: red gaze CSV (bottom-left).")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    out = os.path.join(os.path.abspath(args.output_dir), "gaze_coordinates.csv")
    stats = process_video_ppt(os.path.abspath(args.video), out)
    print("  Wrote", stats["total_frames"], "rows to", stats["csv_path"])


if __name__ == "__main__":
    main()
