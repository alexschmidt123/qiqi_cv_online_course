"""
Step 2: Assign section names (ALL CAPS headings above gaze) and export full text per video.
Uses time buckets (e.g. 1s): run OCR once per bucket on one frame for headings/sections.

The scrollable article rectangle is estimated from OCR on the first time bucket that has
merged headings, then reused unchanged for the entire video (fixed UI; only page content scrolls).

Input: video + gaze CSV from step_1_detect_gaze.py. Output: full_text.txt, gaze_with_section.csv, gaze_ocr_log.txt in --output-dir.

Runs alone: python script/step_2_detect_section.py --video path.mp4 --gaze-csv output/NAME/gaze_coordinates.csv --output-dir output/NAME [--interval 1]
"""
import argparse
import csv
import os
from typing import List, Tuple, Optional, Dict, Any, Set

# Type for one frame-with-gaze in a bucket: (frame_idx, ts, x_bl, y_bl)
BucketFrame = Tuple[int, float, Optional[float], Optional[float]]

import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None

DEFAULT_INTERVAL_SEC = 1.0

# Fallback article bounds when OCR cannot estimate the scrollable article box.
ARTICLE_BOTTOM_MIN_BL = 300
ARTICLE_TOP_OFFSET_PX = 80  # from top of frame (nav); article_ymax_bl = h - 1 - ARTICLE_TOP_OFFSET_PX
# Minimum left inset (fraction of width) when headings are missing (fallback only).
ARTICLE_LEFT_MIN_FRAC = 0.15
# Distance from frame right edge to the left edge of the scrollbar (scrollable pane ends here).
SCROLLBAR_WIDTH_PX = 22
# Gap between main title bottom and the top of the scrollable rectangle.
TITLE_TO_ARTICLE_GAP_PX = 4
# Main title only when gaze is strictly above the first heading's top edge (not overlapping first section).
MAIN_TITLE_STRICT_TOL_PX = 5  # small tolerance for OCR bbox noise
SECTION_MAIN_TITLE = "main title"


def _load_gaze_csv(path: str) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
    rows: List[Tuple[int, float, Optional[float], Optional[float]]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            try:
                idx = int(row.get("frame_idx", i))
            except (ValueError, TypeError):
                idx = i
            try:
                ts = float(row.get("timestamp_sec", 0.0))
            except (ValueError, TypeError):
                ts = 0.0
            x_str = (row.get("x_bl") or "").strip()
            y_str = (row.get("y_bl") or "").strip()
            x_bl = float(x_str) if x_str else None
            y_bl = float(y_str) if y_str else None
            rows.append((idx, ts, x_bl, y_bl))
    return rows


def bl_to_tl(x_bl: Optional[float], y_bl: Optional[float], frame_height: int) -> Tuple[Optional[float], Optional[float]]:
    if frame_height <= 0 or y_bl is None:
        return x_bl, None
    y_tl = (frame_height - 1) - y_bl
    return x_bl, y_tl


def _is_all_caps(text: str) -> bool:
    if not text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 2:
        return False
    return all(ch.isupper() for ch in letters)


def _ocr_frame(reader: "easyocr.Reader", frame: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
    result = reader.readtext(frame)
    out: List[Tuple[np.ndarray, str, float]] = []
    for (bbox, text, conf) in result:
        if not text:
            continue
        try:
            c = float(conf or 0.0)
        except Exception:
            c = 0.0
        if c < 0.03:
            continue
        txt = str(text).strip()
        if not txt:
            continue
        try:
            arr = np.array(bbox, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
        except Exception:
            continue
        out.append((arr, txt, c))
    return out


def _frame_headings_and_text(
    ocr_result: List[Tuple[np.ndarray, str, float]]
) -> Tuple[List[Tuple[float, float, float, float, str]], List[str]]:
    headings: List[Tuple[float, float, float, float, str]] = []
    texts: List[str] = []
    for bbox, txt, _ in ocr_result:
        texts.append(txt)
        if _is_all_caps(txt):
            ys = bbox[:, 1]
            xs = bbox[:, 0]
            y_top = float(min(ys))
            y_bottom = float(max(ys))
            x_min = float(min(xs))
            x_max = float(max(xs))
            headings.append((y_bottom, y_top, x_min, x_max, txt))
    headings.sort(key=lambda h: h[0])
    return headings, texts


def _merge_same_line_headings(
    headings: List[Tuple[float, float, float, float, str]]
) -> List[Tuple[float, float, float, float, str]]:
    """
    Merge ALL-CAPS fragments OCR split on one row (e.g. 'STRL' + 'GTHS') into one heading.
    """
    if len(headings) <= 1:
        return headings
    by_line = sorted(headings, key=lambda h: (h[1], h[2]))
    merged: List[Tuple[float, float, float, float, str]] = []
    i = 0
    while i < len(by_line):
        yb, yt, xmin, xmax, txt = by_line[i]
        j = i + 1
        while j < len(by_line):
            yb2, yt2, xmin2, xmax2, txt2 = by_line[j]
            if abs(yt - yt2) <= 12 and abs(yb - yb2) <= 12:
                xmin = min(xmin, xmin2)
                xmax = max(xmax, xmax2)
                yt = min(yt, yt2)
                yb = max(yb, yb2)
                txt = f"{txt} {txt2}".strip()
                j += 1
            else:
                break
        merged.append((yb, yt, xmin, xmax, txt))
        i = j
    merged.sort(key=lambda h: h[0])
    return merged


def _clamp_article_box(
    box: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, float, float]:
    """
    Clip article box to the frame and cap the bottom so the box does not extend into the
    operation bar. Do NOT force a minimum y0 (that was overriding OCR and mis-drew the top edge).
    """
    x0, y0, x1, y1 = box
    y_max_safe = float(max(ARTICLE_TOP_OFFSET_PX + 1, frame_h - 1 - ARTICLE_BOTTOM_MIN_BL))
    y1 = min(y1, y_max_safe)
    if y1 < y0:
        y1 = y0
    x0 = max(0.0, min(x0, float(frame_w - 1)))
    x1 = max(0.0, min(x1, float(frame_w - 1)))
    if x1 < x0:
        x1 = x0
    return x0, y0, x1, y1


def _is_ui_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    ui_tokens = [
        "about me", "research", "teaching", "professional experience", "portfolio",
        "replay", "copyright", "http", "https", "jenniferlu", "st jude"
    ]
    return any(tok in t for tok in ui_tokens)


def _bbox_tl(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    xs = bbox[:, 0]
    ys = bbox[:, 1]
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))


def _main_title_bottom_y_tl(
    ocr_result: List[Tuple[np.ndarray, str, float]],
    headings: List[Tuple[float, float, float, float, str]],
    frame_w: int,
) -> Optional[float]:
    """
    Bottom edge of main title in top-left y (largest y of the title block above first section heading).
    Skips ALL-CAPS lines that look like section headings; allows long ALL-CAPS title lines.
    """
    if not headings:
        return None
    first_heading_y_top = headings[0][1]
    col_lo = 0.10 * frame_w
    col_hi = 0.92 * frame_w
    primary_bottoms: List[float] = []
    soft_bottoms: List[float] = []

    for bbox, txt, _ in ocr_result:
        if _is_ui_text(txt):
            continue
        x0, y0, x1, y1 = _bbox_tl(bbox)
        if (x1 - x0) < 36 or (y1 - y0) < 6:
            continue
        # Must lie above the first ALL-CAPS section heading
        if y1 >= first_heading_y_top - 2:
            continue
        # Roughly in article column (layout differs by video; loose bounds)
        if x1 < col_lo or x0 > col_hi:
            continue
        t = (txt or "").strip()
        if not _is_all_caps(t):
            primary_bottoms.append(y1)
        elif len(t) >= 14 or len(t.split()) >= 3:
            # Long ALL-CAPS line could be a main title on some pages
            soft_bottoms.append(y1)

    if primary_bottoms:
        return max(primary_bottoms)
    if soft_bottoms:
        return max(soft_bottoms)
    return None


def _estimate_article_box(
    ocr_result: List[Tuple[np.ndarray, str, float]],
    headings: List[Tuple[float, float, float, float, str]],
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, float, float]:
    """
    Scrollable article rectangle in top-left coords (x_min, y_min, x_max, y_max).

    Rules (layout varies by video; all from OCR + same bottom rule as gaze gating):
    - Left: left edge of ALL-CAPS section headings (min of heading x_min).
    - Top: just below bottom edge of main title; if no title detected, top of first heading.
    - Right: left edge of scrollbar ≈ min(frame_w - scrollbar_width, rightmost content + margin).
    - Bottom: top of red operation panel strip (same as ARTICLE_BOTTOM_MIN_BL in clamp).
    """
    xs_min: List[float] = []
    ys_min: List[float] = []
    xs_max: List[float] = []
    ys_max: List[float] = []

    for bbox, txt, _ in ocr_result:
        if _is_ui_text(txt):
            continue
        x0, y0, x1, y1 = _bbox_tl(bbox)
        if (x1 - x0) < 30 or (y1 - y0) < 8:
            continue
        xs_min.append(x0)
        ys_min.append(y0)
        xs_max.append(x1)
        ys_max.append(y1)

    # --- Left: left edge of section names (ALL-CAPS headings) ---
    if headings:
        x_min = min(h[2] for h in headings)
    elif xs_min:
        x_min = max(float(ARTICLE_LEFT_MIN_FRAC * frame_w), min(xs_min) - 20.0)
    else:
        x_min = float(ARTICLE_LEFT_MIN_FRAC * frame_w)

    # --- Top: below bottom of main title, else top of first heading ---
    mt_bottom = _main_title_bottom_y_tl(ocr_result, headings, frame_w)
    if mt_bottom is not None:
        y_min = mt_bottom + float(TITLE_TO_ARTICLE_GAP_PX)
    elif headings:
        y_min = headings[0][1]
    elif ys_min:
        y_min = max(0.0, min(ys_min) - 20.0)
    else:
        y_min = float(ARTICLE_TOP_OFFSET_PX)

    # --- Right: left edge of scrollbar; use widest content (headings + body) ---
    scrollbar_left_x = float(frame_w - 1 - SCROLLBAR_WIDTH_PX)
    content_right_candidates: List[float] = list(xs_max)
    if headings:
        content_right_candidates.extend(h[3] for h in headings)
    if content_right_candidates:
        content_right = max(content_right_candidates) + 20.0
        x_max = min(scrollbar_left_x, content_right)
    else:
        x_max = min(scrollbar_left_x, 0.90 * float(frame_w))

    # --- Bottom: same as before (clamp will also cap to above operation bar) ---
    if ys_max:
        y_max = min(float(frame_h - 1), max(ys_max) + 20.0)
    else:
        y_max = float(frame_h - 1)

    return _clamp_article_box((x_min, y_min, x_max, y_max), frame_w, frame_h)


def _pick_unified_article_box(
    article_box_per_bucket: Dict[int, Tuple[float, float, float, float]],
    sorted_buckets: List[int],
    headings_per_bucket: Dict[int, List[Tuple[float, float, float, float, str]]],
) -> Optional[Tuple[float, float, float, float]]:
    """
    One scrollable rectangle for the entire video (layout is fixed; scrolling only moves content).

    Prefer the first time bucket (chronological) that has both an article box and merged headings.
    Otherwise use the first bucket with any box.
    """
    for b in sorted_buckets:
        if b in article_box_per_bucket and headings_per_bucket.get(b):
            return article_box_per_bucket[b]
    for b in sorted_buckets:
        if b in article_box_per_bucket:
            return article_box_per_bucket[b]
    return None


def _refresh_debug_rows_unified_article(
    debug_bucket_rows: List[Tuple[Any, ...]],
    unified: Tuple[float, float, float, float],
    h_video: int,
    first_heading_y_per_bucket: Dict[int, float],
) -> List[Tuple[Any, ...]]:
    """Replace article box columns and recompute in_article / main_title_rule for unified geometry."""
    bx0, by0, bx1, by1 = unified
    new_rows: List[Tuple[Any, ...]] = []
    for row in debug_bucket_rows:
        if len(row) < 15:
            new_rows.append(row)
            continue
        bucket = int(row[0])
        x_bl_s, y_bl_s = row[3], row[4]
        x_tl_s, y_tl_s = row[5], row[6]
        all_caps_str = row[14]
        try:
            y_bl = float(y_bl_s) if str(y_bl_s).strip() != "" else None
        except (TypeError, ValueError):
            y_bl = None
        try:
            x_tl = float(x_tl_s) if str(x_tl_s).strip() != "" else None
            y_tl = float(y_tl_s) if str(y_tl_s).strip() != "" else None
        except (TypeError, ValueError):
            x_tl, y_tl = None, None
        in_article = 0
        main_title_rule = 0
        if x_tl is not None and y_tl is not None:
            in_article = int(
                bx0 <= float(x_tl) <= bx1 and by0 <= float(y_tl) <= by1
            )
            if in_article and y_bl is not None:
                y_bl_cur = float(y_bl)
                in_vertical_article_band = (
                    y_bl_cur >= ARTICLE_BOTTOM_MIN_BL
                    and y_bl_cur <= (h_video - 1 - ARTICLE_TOP_OFFSET_PX)
                )
                if in_vertical_article_band:
                    first_y = first_heading_y_per_bucket.get(bucket)
                    if first_y is not None and float(y_tl) < first_y - MAIN_TITLE_STRICT_TOL_PX:
                        main_title_rule = 1
        new_rows.append(
            (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                bx0,
                by0,
                bx1,
                by1,
                in_article,
                main_title_rule,
                row[13],
                all_caps_str,
            )
        )
    return new_rows


def _choose_heading_for_gaze(
    headings: List[Tuple[float, float, float, float, str]],
    gaze_x_tl: float,
    gaze_y_tl: float,
    x_margin: float = 40.0,
) -> str:
    # Pass 1: prefer x-overlap with heading column.
    best: Optional[Tuple[float, float, float, float, str]] = None
    best_delta = None
    for (y_bottom, y_top, x_min, x_max, txt) in headings:
        if y_bottom > gaze_y_tl:
            continue
        if gaze_x_tl < x_min - x_margin or gaze_x_tl > x_max + x_margin:
            continue
        delta = gaze_y_tl - y_bottom
        if best is None or delta < best_delta:  # type: ignore[arg-type]
            best = (y_bottom, y_top, x_min, x_max, txt)
            best_delta = delta
    if best is not None:
        return best[4]

    # Pass 1b: gaze inside heading bbox (e.g. reading the heading line; OCR splits can break pass 1).
    for (y_bottom, y_top, x_min, x_max, txt) in headings:
        if y_top <= gaze_y_tl <= y_bottom:
            if gaze_x_tl >= x_min - x_margin and gaze_x_tl <= x_max + x_margin:
                return txt

    # Pass 2 (fallback): nearest heading above by y only.
    best = None
    best_delta = None
    for (y_bottom, y_top, x_min, x_max, txt) in headings:
        if y_bottom > gaze_y_tl:
            continue
        delta = gaze_y_tl - y_bottom
        if best is None or delta < best_delta:  # type: ignore[arg-type]
            best = (y_bottom, y_top, x_min, x_max, txt)
            best_delta = delta
    if best is not None:
        return best[4]

    # Pass 3: very wide x margin (narrow columns / split OCR).
    wide = max(x_margin * 4, 200.0)
    best = None
    best_delta = None
    for (y_bottom, y_top, x_min, x_max, txt) in headings:
        if y_bottom > gaze_y_tl:
            continue
        if gaze_x_tl < x_min - wide or gaze_x_tl > x_max + wide:
            continue
        delta = gaze_y_tl - y_bottom
        if best is None or delta < best_delta:  # type: ignore[arg-type]
            best = (y_bottom, y_top, x_min, x_max, txt)
            best_delta = delta
    return best[4] if best is not None else ""


def process_video(
    video_path: str, gaze_csv_path: str, output_dir: str, interval_sec: float = DEFAULT_INTERVAL_SEC
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_bucket_csv = os.path.join(debug_dir, "step2_bucket_debug.csv")
    if easyocr is None:
        raise RuntimeError("easyocr is required; install with `pip install easyocr`.")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

    gaze_rows = _load_gaze_csv(gaze_csv_path)
    if not gaze_rows:
        raise RuntimeError(f"No rows in gaze CSV: {gaze_csv_path}")

    # Bucket = int(ts / interval_sec). Per bucket, list all frames with gaze (sorted by frame_idx).
    # We'll try frames in order until a read succeeds, without leaving the bucket.
    bucket_to_frames: Dict[int, List[BucketFrame]] = {}
    for (frame_idx, ts, x_bl, y_bl) in gaze_rows:
        if x_bl is None or y_bl is None:
            continue
        bucket = int(ts / interval_sec)
        bucket_to_frames.setdefault(bucket, []).append((frame_idx, ts, x_bl, y_bl))
    for b in bucket_to_frames:
        bucket_to_frames[b].sort(key=lambda r: r[0])  # by frame_idx

    # Run OCR only for those buckets (one frame per bucket; try next frame if read fails)
    section_per_bucket: Dict[int, str] = {}
    article_box_per_bucket: Dict[int, Tuple[float, float, float, float]] = {}
    first_heading_y_per_bucket: Dict[int, float] = {}
    headings_per_bucket: Dict[int, List[Tuple[float, float, float, float, str]]] = {}
    transcript_seen: Set[str] = set()
    transcript_order: List[str] = []
    ocr_log_path = os.path.join(output_dir, "gaze_ocr_log.txt")
    gaze_with_section_rows: List[Tuple[int, float, str, str, str]] = []
    debug_bucket_rows: List[Tuple[Any, ...]] = []
    sorted_buckets = sorted(bucket_to_frames.keys()) if bucket_to_frames else []

    if not sorted_buckets:
        # No gaze in any bucket; still write empty section for all rows
        gaze_with_section_rows = [
            (frame_idx, ts, str(x_bl or ""), str(y_bl or ""), "")
            for (frame_idx, ts, x_bl, y_bl) in gaze_rows
        ]
        with open(ocr_log_path, "w", encoding="utf-8") as log_f:
            log_f.write("# frame_idx\ttimestamp_sec\tx_bl\ty_bl\tsection\tall_caps_candidates\n")
        with open(debug_bucket_csv, "w", newline="", encoding="utf-8") as fdbg:
            wdbg = csv.writer(fdbg)
            wdbg.writerow([
                "bucket", "frame_idx", "timestamp_sec", "gaze_x_bl", "gaze_y_bl",
                "gaze_x_tl", "gaze_y_tl", "article_x0_tl", "article_y0_tl",
                "article_x1_tl", "article_y1_tl", "in_article", "main_title_rule",
                "section", "all_caps_candidates"
            ])
        print("  Step 2 done: 0 OCR runs (no gaze frames), {} output rows.".format(len(gaze_with_section_rows)))
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        print("  Loading EasyOCR (one-time, can take a moment)...")
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        section_per_bucket: Dict[int, str] = {}
        with open(ocr_log_path, "w", encoding="utf-8") as log_f:
            log_f.write("# frame_idx\ttimestamp_sec\tx_bl\ty_bl\tsection\tall_caps_candidates\n")
            for run_idx, bucket in enumerate(sorted_buckets):
                frames_in_bucket = bucket_to_frames[bucket]
                # Try frames with gaze in order until read succeeds; never go into next bucket
                frame_idx, ts, x_bl, y_bl = frames_in_bucket[0]
                frame = None
                for (fidx, t, xb, yb) in frames_in_bucket:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_idx, ts, x_bl, y_bl = fidx, t, xb, yb
                        break
                if frame is None:
                    section_per_bucket[bucket] = "none"
                    headings_per_bucket[bucket] = []
                    article_box_per_bucket[bucket] = _clamp_article_box(
                        (
                            0.15 * w_video,
                            float(ARTICLE_TOP_OFFSET_PX),
                            0.90 * w_video,
                            float(h_video - 1 - ARTICLE_BOTTOM_MIN_BL),
                        ),
                        w_video,
                        h_video,
                    )
                    debug_bucket_rows.append(
                        (bucket, frame_idx, ts, x_bl, y_bl, "", "", "", "", "", "", 0, 0, "none", "")
                    )
                    continue
                if run_idx <= 2 or (run_idx + 1) % 20 == 0:
                    print("  OCR bucket {} at {:.1f}s (run #{})".format(bucket, ts, run_idx + 1))
                h, w = frame.shape[:2]
                ocr_result = _ocr_frame(reader, frame)
                headings, texts = _frame_headings_and_text(ocr_result)
                headings = _merge_same_line_headings(headings)
                headings_per_bucket[bucket] = list(headings)
                article_x0, article_y0, article_x1, article_y1 = _estimate_article_box(
                    ocr_result, headings, w, h
                )
                article_box_per_bucket[bucket] = (article_x0, article_y0, article_x1, article_y1)
                if headings:
                    first_heading_y_per_bucket[bucket] = headings[0][1]
                for t in texts:
                    if t not in transcript_seen:
                        transcript_seen.add(t)
                        transcript_order.append(t)
                x_tl, y_tl = bl_to_tl(x_bl, y_bl, h)
                current_section = "none"
                in_article = False
                main_title_rule = False
                if x_tl is not None and y_tl is not None:
                    y_bl_cur = float(y_bl) if y_bl is not None else -1.0
                    in_vertical_article_band = (
                        y_bl_cur >= ARTICLE_BOTTOM_MIN_BL
                        and y_bl_cur <= (h - 1 - ARTICLE_TOP_OFFSET_PX)
                    )
                    in_article = (
                        article_x0 <= float(x_tl) <= article_x1
                        and article_y0 <= float(y_tl) <= article_y1
                    )
                    if in_article and in_vertical_article_band:
                        # Main title only when gaze is strictly above first heading's top (not first section).
                        if headings and float(y_tl) < headings[0][1] - MAIN_TITLE_STRICT_TOL_PX:
                            current_section = SECTION_MAIN_TITLE
                            main_title_rule = True
                        else:
                            current_section = _choose_heading_for_gaze(headings, float(x_tl), float(y_tl))
                        if not current_section:
                            current_section = "none"
                    else:
                        current_section = "none"
                section_per_bucket[bucket] = current_section
                all_caps_str = "; ".join(hh[4] for hh in headings)
                debug_bucket_rows.append(
                    (
                        bucket, frame_idx, ts, x_bl, y_bl, x_tl if x_tl is not None else "",
                        y_tl if y_tl is not None else "", article_x0, article_y0, article_x1,
                        article_y1, 1 if in_article else 0, 1 if main_title_rule else 0,
                        current_section, all_caps_str
                    )
                )
                log_f.write(
                    "{}\t{:.4f}\t{}\t{}\t{}\t{}\n".format(
                        frame_idx, ts, x_bl, y_bl, current_section, all_caps_str,
                    )
                )
        cap.release()

        # Single scrollable rectangle for the whole video (UI chrome is fixed; only page content scrolls).
        unified_article = _pick_unified_article_box(
            article_box_per_bucket, sorted_buckets, headings_per_bucket
        )
        if unified_article is not None:
            for b in sorted_buckets:
                article_box_per_bucket[b] = unified_article
            debug_bucket_rows = _refresh_debug_rows_unified_article(
                debug_bucket_rows, unified_article, h_video, first_heading_y_per_bucket
            )

        # Build gaze_with_section_rows with per-frame gating:
        # - outside bucket article box -> none
        # - above first heading in article -> main title
        # - else use bucket section label, OR if bucket label is "none" recompute heading per frame
        #   (bucket "none" is truthy in Python, so we must not use `elif not section:` for that case.)
        gaze_with_section_rows = []
        for (frame_idx, ts, x_bl, y_bl) in gaze_rows:
            x_str = str(x_bl) if x_bl is not None else ""
            y_str = str(y_bl) if y_bl is not None else ""
            bucket = int(ts / interval_sec)
            section = section_per_bucket.get(bucket, "none")
            if x_bl is None or y_bl is None:
                section = "none"
            else:
                x_tl, y_tl = bl_to_tl(x_bl, y_bl, h_video)
                box = article_box_per_bucket.get(bucket)
                if x_tl is None or y_tl is None or box is None:
                    section = "none"
                else:
                    bx0, by0, bx1, by1 = box
                    in_article = bx0 <= float(x_tl) <= bx1 and by0 <= float(y_tl) <= by1
                    in_vertical_article_band = (
                        float(y_bl) >= ARTICLE_BOTTOM_MIN_BL
                        and float(y_bl) <= (h_video - 1 - ARTICLE_TOP_OFFSET_PX)
                    )
                    if not in_article or not in_vertical_article_band:
                        section = "none"
                    else:
                        first_y = first_heading_y_per_bucket.get(bucket)
                        if first_y is not None and float(y_tl) < first_y - MAIN_TITLE_STRICT_TOL_PX:
                            section = SECTION_MAIN_TITLE
                        else:
                            sb = (section or "").strip()
                            if sb and sb.lower() != "none":
                                section = sb
                            else:
                                hlist = headings_per_bucket.get(bucket, [])
                                section = (
                                    _choose_heading_for_gaze(hlist, float(x_tl), float(y_tl))
                                    if hlist
                                    else "none"
                                )
                                if not section:
                                    section = "none"
            gaze_with_section_rows.append((frame_idx, ts, x_str, y_str, section))

        print("  Step 2 done: {} OCR runs (1 per {}s bucket), {} output rows.".format(
            len(sorted_buckets), interval_sec, len(gaze_with_section_rows)))
        with open(debug_bucket_csv, "w", newline="", encoding="utf-8") as fdbg:
            wdbg = csv.writer(fdbg)
            wdbg.writerow([
                "bucket", "frame_idx", "timestamp_sec", "gaze_x_bl", "gaze_y_bl",
                "gaze_x_tl", "gaze_y_tl", "article_x0_tl", "article_y0_tl",
                "article_x1_tl", "article_y1_tl", "in_article", "main_title_rule",
                "section", "all_caps_candidates"
            ])
            for row in debug_bucket_rows:
                wdbg.writerow(row)

    transcript_path = os.path.join(output_dir, "full_text.txt")
    with open(transcript_path, "w", encoding="utf-8") as tf:
        for line in transcript_order:
            tf.write(line + "\n")

    out_csv = os.path.join(output_dir, "gaze_with_section.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame_idx", "timestamp_sec", "x_bl", "y_bl", "section"])
        for r in gaze_with_section_rows:
            wcsv.writerow(r)

    return {
        "video": video_path,
        "gaze_csv": gaze_csv_path,
        "output_dir": output_dir,
        "transcript_path": transcript_path,
        "gaze_with_section_csv": out_csv,
        "ocr_log_path": ocr_log_path,
        "debug_bucket_csv": debug_bucket_csv,
        "frames_processed": len(gaze_with_section_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign section (ALL CAPS heading) to gaze; export full text.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--gaze-csv", type=str, required=True, help="Gaze coordinates CSV from step_1.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder for this video.")
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        metavar="SEC",
        help="Time bucket in seconds; OCR once per bucket (default: 1.0).",
    )
    args = parser.parse_args()

    stats = process_video(args.video, args.gaze_csv, args.output_dir, interval_sec=args.interval)
    print("Video:", stats["video"])
    print("Gaze CSV:", stats["gaze_csv"])
    print("Output dir:", stats["output_dir"])
    print("Frames processed:", stats["frames_processed"])
    print("Transcript:", stats["transcript_path"])
    print("Gaze+section CSV:", stats["gaze_with_section_csv"])
    print("Gaze OCR log:", stats["ocr_log_path"])
    print("Debug bucket CSV:", stats["debug_bucket_csv"])


if __name__ == "__main__":
    main()
