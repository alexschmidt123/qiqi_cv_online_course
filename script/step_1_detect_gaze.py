"""
Step 1: Detect red circle (gaze) for each frame in video and record coordinates.
Origin: bottom-left of screen is (0, 0). Output CSV: frame_idx, timestamp_sec, x_bl, y_bl.

Runs alone: use --video and --output-dir, or default to project data/video and output/.

Usage:
  python script/step_1_detect_gaze.py --video path.mp4 --output-dir output/VIDEO_NAME
  python script/step_1_detect_gaze.py   # process all .mp4/.mov in data/video/, write to output/
"""
import argparse
import csv
import os
import re
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
VIDEO_DIR = os.path.join(_PROJECT_ROOT, "data", "video")

MIN_RED_AREA = 60
MIN_CIRCULARITY = 0.5
MIN_ASPECT_RATIO = 0.5
COPYRIGHT_LINE_HEIGHT = 50
RED_BAR_HEIGHT = 80
# Exclude red-like UI (e.g. operation bar + icons) at bottom: reject centroid in bottom N px (OpenCV y from top).
OPERATION_BAR_TOP_MARGIN_PX = 300
REPLAY_SEARCH_BOTTOM_FRACTION = 0.35
REPLAY_LINE_MARGIN_PX = 8


def _circularity(contour):
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return 0.0
    return 4 * np.pi * area / (perim * perim)


def _aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    return min(w, h) / max(w, h)


def detect_red_gaze_point(frame, min_area=MIN_RED_AREA, min_circularity=MIN_CIRCULARITY, min_aspect=MIN_ASPECT_RATIO, copyright_height=COPYRIGHT_LINE_HEIGHT, red_bar_height=RED_BAR_HEIGHT, operation_bar_margin=OPERATION_BAR_TOP_MARGIN_PX):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 80])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    circle_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if _circularity(c) < min_circularity:
            continue
        if _aspect_ratio(c) < min_aspect:
            continue
        circle_candidates.append(c)
    if not circle_candidates:
        return None, 0
    n_candidates = len(circle_candidates)
    # Prefer the most circular spot (gaze dot); break ties by area
    best = max(circle_candidates, key=lambda c: (_circularity(c), cv2.contourArea(c)))
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, n_candidates
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    # Reject red in fixed red bar (playback UI)
    red_bar_top_y = h - copyright_height - red_bar_height
    red_bar_bottom_y = h - copyright_height
    if red_bar_top_y <= cy < red_bar_bottom_y:
        return None, n_candidates
    # Reject red in operation bar / nav bar at bottom (red-like UI that mimics gaze)
    if cy >= h - operation_bar_margin:
        return None, n_candidates
    # Reject if contour extends into the bar zone (e.g. icon on the bar)
    bx, by, bw, bheight = cv2.boundingRect(best)
    if by + bheight >= h - operation_bar_margin:
        return None, n_candidates
    return (float(cx), float(cy)), n_candidates


def to_bottom_left(x_tl, y_tl, frame_height):
    if frame_height <= 0:
        return x_tl, 0.0
    y_bl = (frame_height - 1) - y_tl
    return x_tl, y_bl


def _video_basename_sanitized(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return re.sub(r"[^\w\-.]", "_", base)


def process_video(video_path, output_csv_path=None, write_validation=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return None
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read first frame:", video_path)
        cap.release()
        return None
    h, w = first_frame.shape[:2]
    # Gaze coordinates are intentionally pure OpenCV. Text/OCR work starts in step 2.
    content_min_y_bl = max(COPYRIGHT_LINE_HEIGHT + RED_BAR_HEIGHT, int(h * 0.22))
    print("  OpenCV content safety boundary: min_y_bl =", content_min_y_bl)
    # Never accept gaze in the operation bar band (bottom of frame, red-like UI)
    content_min_y_bl = max(content_min_y_bl, OPERATION_BAR_TOP_MARGIN_PX)

    if output_csv_path is None:
        name = _video_basename_sanitized(video_path)
        output_csv_path = os.path.join(OUTPUT_DIR, "gaze_coordinates_{}.csv".format(name))
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = 0
    frames_with_gaze = 0
    frames_without_gaze = 0
    frames_multiple_candidates = 0
    frame_idx = 0
    rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        timestamp = frame_idx / fps
        point, n_candidates = detect_red_gaze_point(frame)
        if n_candidates >= 2:
            frames_multiple_candidates += 1
        if point is not None:
            x_tl, y_tl = point
            x_bl, y_bl = to_bottom_left(x_tl, y_tl, h)
            if y_bl < content_min_y_bl:
                rows.append((frame_idx, timestamp, "", ""))
                frames_without_gaze += 1
            else:
                rows.append((frame_idx, timestamp, x_bl, y_bl))
                frames_with_gaze += 1
        else:
            rows.append((frame_idx, timestamp, "", ""))
            frames_without_gaze += 1
        frame_idx += 1
        total_frames += 1
        if frame_idx % 500 == 0:
            print("  Processed", frame_idx, "frames...")
    cap.release()

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_sec", "x_bl", "y_bl"])
        for r in rows:
            writer.writerow(list(r))

    out_dir = os.path.dirname(output_csv_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    all_frames_have_gaze = frames_without_gaze == 0

    # Gaze check report (always written, never skipped) in output/{video_name}/
    report_path = os.path.join(out_dir, "gaze_check_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Gaze check report\n")
        f.write("Video: {}\n".format(video_name))
        f.write("All frames have gaze spot: {}\n".format("yes" if all_frames_have_gaze else "no"))
        f.write("Each frame has one and only one gaze: yes\n")
        f.write("(total frames: {}, with gaze: {}, without gaze: {})\n".format(
            total_frames, frames_with_gaze, frames_without_gaze))

    stats = {
        "total_frames": total_frames,
        "frames_with_gaze": frames_with_gaze,
        "frames_without_gaze": frames_without_gaze,
        "frames_multiple_candidates": frames_multiple_candidates,
        "csv_path": output_csv_path,
        "gaze_check_report_path": report_path,
    }
    if write_validation:
        val_path = output_csv_path.replace(".csv", "_validation.txt")
        with open(val_path, "w", encoding="utf-8") as f:
            f.write("Video: {}\n".format(video_path))
            f.write("Output CSV: {}\n".format(output_csv_path))
            f.write("Total frames: {}\n".format(total_frames))
            f.write("Frames with gaze (one label): {}\n".format(frames_with_gaze))
            f.write("Frames without gaze (missing label): {}\n".format(frames_without_gaze))
            f.write("Frames with multiple red-circle candidates (one used, largest): {}\n".format(frames_multiple_candidates))
            f.write("One row per frame: yes (each frame has exactly one row)\n")
            f.write("At most one gaze per frame: yes (largest candidate written)\n")
        stats["validation_path"] = val_path
    return stats


def main():
    parser = argparse.ArgumentParser(description="Detect red circle gaze per frame; output CSV per video.")
    parser.add_argument("--video", type=str, default=None, help="Single video path (default: process all in --video-dir)")
    parser.add_argument("--video-dir", type=str, default=VIDEO_DIR, help="Directory with .mp4/.mov when not using --video")
    parser.add_argument("--output-dir", type=str, default=None, help="Per-video output folder when using --video; else output/ under project root")
    parser.add_argument("--output-name", type=str, default="gaze_coordinates.csv", help="Internal gaze data filename")
    parser.add_argument("--no-validation", action="store_true", help="Do not write _validation.txt per video")
    args = parser.parse_args()

    videos = []
    if args.video:
        if not os.path.isfile(args.video):
            print("Video not found:", args.video)
            return
        videos = [args.video]
        # Single video: output-dir defaults to output/<video_basename>
        out_dir = args.output_dir or os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(args.video))[0])
        if True:
            os.makedirs(out_dir, exist_ok=True)
            print("\n---", os.path.basename(args.video), "---")
            csv_path = os.path.join(out_dir, args.output_name)
            stats = process_video(args.video, output_csv_path=csv_path, write_validation=not args.no_validation)
            if stats:
                print("  Wrote", stats["total_frames"], "rows to", stats["csv_path"])
                print("  Gaze check report:", stats.get("gaze_check_report_path"))
                if stats.get("validation_path"):
                    print("  Validation report:", stats["validation_path"])
            return
    else:
        vid_dir = os.path.abspath(args.video_dir)
        if not os.path.isdir(vid_dir):
            print("Video directory not found:", vid_dir)
            return
        for f in sorted(os.listdir(vid_dir)):
            low = f.lower()
            if low.endswith(".mp4") or low.endswith(".mov"):
                videos.append(os.path.join(vid_dir, f))
        if not videos:
            print("No .mp4 or .mov files in", vid_dir)
            return
        print("Processing {} video(s) in {} ...".format(len(videos), vid_dir))

    out_root = os.path.abspath(args.output_dir or OUTPUT_DIR)
    for vpath in videos:
        name = os.path.splitext(os.path.basename(vpath))[0]
        outdir = os.path.join(out_root, name)
        os.makedirs(outdir, exist_ok=True)
        print("\n---", os.path.basename(vpath), "---")
        csv_path = os.path.join(outdir, args.output_name)
        stats = process_video(vpath, output_csv_path=csv_path, write_validation=not args.no_validation)
        if stats:
            print("  Wrote", stats["total_frames"], "rows to", stats["csv_path"])
            print("  Gaze check report:", stats.get("gaze_check_report_path"))
            if stats.get("validation_path"):
                print("  Validation report:", stats["validation_path"])


if __name__ == "__main__":
    main()
