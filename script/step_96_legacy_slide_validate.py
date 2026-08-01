"""
Step 4 (PPT): 5 validation frames — original + overlay (same layout each).

Draws: left/right **black pillarbox** strips and **red** nav, bright-blue **PPT** outline,
**orange** rects for `image`, **sky-blue** circles for **buttons** (1–4, i, !),
**violet** rect for **`button_text`**, **green** for **`heading`**, **light-blue** for **`paragraph`**,
**magenta** outline for **`navigation_bar`** (detector bbox), gaze dot, and label.
Uses the same detectors as step 2 when EasyOCR is available.

**PPT rect for each validation frame** comes from **`detect_ppt_rect_from_chrome(frame)`**
when it succeeds (same as pipeline chrome detection). Falls back to `debug/ppt_region.txt`
only if chrome fails — stale file rects misplace slide edges and break pillar walks.

Reads final_gaze_table.txt (tab) or final CSV; optional fallback `debug/ppt_region.txt`.

Usage:
  python scripts_image/step_4_validate_image.py --video path.mp4 --final-table .../final_gaze_table.txt --output-dir ...
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

_SI_DIR = os.path.dirname(os.path.abspath(__file__))
if _SI_DIR not in sys.path:
    sys.path.insert(0, _SI_DIR)
from step_94_legacy_slide_detect_elements import (
    Element,
    detect_elements_for_validation,
    detect_text_frame_popup_rect,
)

from step_80_chrome_geometry import (
    NAV_BAR_HEIGHT_PX,
    detect_nav_bar_top_y_in_player,
    detect_ppt_rect_from_chrome,
    detect_red_nav_bottom_y,
)

# PPT = tight rect from chrome (inner black-bar edges, nav top); no extra inset here
PPT_BORDER_BGR = (255, 100, 40)
PPT_LABEL_BGR = (255, 200, 120)
# button_text = black-border popup (distinct from wrong old purple-only heuristic)
BUTTON_TEXT_FRAME_BORDER_BGR = (210, 70, 230)
BUTTON_TEXT_FRAME_LABEL_BGR = (220, 200, 255)
IMAGE_BORDER_BGR = (0, 140, 255)  # orange (BGR)
IMAGE_LABEL_BGR = (0, 200, 255)
BUTTON_CIRCLE_BGR = (255, 200, 120)  # sky blue (BGR)
HEADING_BORDER_BGR = (60, 200, 60)  # green (BGR)
HEADING_LABEL_BGR = (120, 255, 120)
PARAGRAPH_BORDER_BGR = (255, 200, 120)  # cyan / light blue (BGR)
PARAGRAPH_LABEL_BGR = (255, 230, 180)
NAV_ELEMENT_BORDER_BGR = (255, 0, 255)  # magenta — detector `navigation_bar` rect
NAV_ELEMENT_LABEL_BGR = (255, 180, 255)
# Chrome outlines: green / yellow (BGR) — visible on black and on red; avoid white
BLACK_BAR_OUTLINE_BGR = (0, 255, 0)
NAV_BAR_OUTLINE_BGR = (0, 255, 255)
BLACK_BAR_LABEL_BGR = (0, 255, 128)
NAV_BAR_LABEL_BGR = (0, 255, 255)
GAZE_COLOR_BGR = (0, 0, 255)  # red circle = gaze (not an element)
LABEL_TEXT_BGR = (0, 255, 255)  # yellow text for readability

_STRIP_BUTTON_INTERNAL = frozenset(
    {"button_1", "button_2", "button_3", "button_4", "button_i", "button_bang"}
)


def _detector_tag(el: Element, name_counts: Counter, seen: defaultdict[str, int]) -> str:
    seen[el.name] += 1
    if name_counts[el.name] > 1:
        return f"{el.name}[{seen[el.name]}]"
    return el.name


def _draw_detection_legend(
    frame: np.ndarray,
    elements: List[Element],
    w: int,
    h: int,
) -> None:
    """Bottom-left: every element name with pixel bbox (for screenshots / QA)."""
    lines = [
        f"{e.name}: ({int(round(e.x0))},{int(round(e.y0))})-({int(round(e.x1))},{int(round(e.y1))})"
        for e in sorted(elements, key=lambda el: (el.name, el.x0, el.y0))
    ]
    max_lines = 32
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (+{len(elements) - max_lines} more)"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thk = 0.34, 1
    line_h = 14
    pad = 8
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, fs, thk)
        max_w = max(max_w, tw)
    block_w = min(w - 12, max_w + pad * 2)
    block_h = min(len(lines) * line_h + pad * 2, h // 2)
    x0, y0 = 6, h - block_h - 6
    cv2.rectangle(frame, (x0, y0), (x0 + block_w, h - 4), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + block_w, h - 4), (80, 80, 80), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 2
        cv2.putText(frame, line, (x0 + pad, y), font, fs, (230, 230, 230), thk, cv2.LINE_AA)


def _draw_validation_detectors(
    frame: np.ndarray,
    elements: List[Element],
    w: int,
    h: int,
) -> None:
    """Draw every detector element: image, button_text, buttons, heading, paragraph, navigation_bar."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts, thk = 0.55, 2
    name_counts = Counter(e.name for e in elements)
    seen: defaultdict[str, int] = defaultdict(int)

    for el in elements:
        if el.name == "image":
            x0 = max(0, min(w - 1, int(round(el.x0))))
            y0 = max(0, min(h - 1, int(round(el.y0))))
            x1 = max(0, min(w - 1, int(round(el.x1))))
            y1 = max(0, min(h - 1, int(round(el.y1))))
            if x1 > x0 + 2 and y1 > y0 + 2:
                cv2.rectangle(frame, (x0, y0), (x1, y1), IMAGE_BORDER_BGR, 2, lineType=cv2.LINE_AA)
                tag = _detector_tag(el, name_counts, seen)
                (tw_i, th_i), _ = cv2.getTextSize(tag, font, ts, thk)
                cv2.putText(
                    frame,
                    tag,
                    (max(0, x0), max(th_i + 2, y0 - 4)),
                    font,
                    ts,
                    IMAGE_LABEL_BGR,
                    thk,
                    cv2.LINE_AA,
                )
        elif el.name == "button_text":
            x0 = max(0, min(w - 1, int(round(el.x0))))
            y0 = max(0, min(h - 1, int(round(el.y0))))
            x1 = max(0, min(w - 1, int(round(el.x1))))
            y1 = max(0, min(h - 1, int(round(el.y1))))
            if x1 > x0 + 2 and y1 > y0 + 2:
                cv2.rectangle(
                    frame,
                    (x0, y0),
                    (x1, y1),
                    BUTTON_TEXT_FRAME_BORDER_BGR,
                    2,
                    lineType=cv2.LINE_AA,
                )
                tag = _detector_tag(el, name_counts, seen)
                (tw_b, th_b), _ = cv2.getTextSize(tag, font, ts, thk)
                cv2.putText(
                    frame,
                    tag,
                    (max(0, x0), max(th_b + 2, y0 - 4)),
                    font,
                    ts,
                    BUTTON_TEXT_FRAME_LABEL_BGR,
                    thk,
                    cv2.LINE_AA,
                )
        elif el.name in _STRIP_BUTTON_INTERNAL:
            cx = int(round((el.x0 + el.x1) * 0.5))
            cy = int(round((el.y0 + el.y1) * 0.5))
            rw = max(4.0, (el.x1 - el.x0) * 0.55)
            rh = max(4.0, (el.y1 - el.y0) * 0.55)
            r = int(round(max(rw, rh)))
            cx = max(r, min(w - 1 - r, cx))
            cy = max(r, min(h - 1 - r, cy))
            cv2.circle(frame, (cx, cy), r, BUTTON_CIRCLE_BGR, 2, lineType=cv2.LINE_AA)
            tag = _detector_tag(el, name_counts, seen)
            (tw_c, th_c), _ = cv2.getTextSize(tag, font, 0.45, 1)
            cv2.putText(
                frame,
                tag,
                (max(0, cx - tw_c // 2), max(th_c + 2, cy - r - 4)),
                font,
                0.45,
                BUTTON_CIRCLE_BGR,
                1,
                cv2.LINE_AA,
            )
        elif el.name == "heading":
            x0 = max(0, min(w - 1, int(round(el.x0))))
            y0 = max(0, min(h - 1, int(round(el.y0))))
            x1 = max(0, min(w - 1, int(round(el.x1))))
            y1 = max(0, min(h - 1, int(round(el.y1))))
            if x1 > x0 + 2 and y1 > y0 + 2:
                cv2.rectangle(frame, (x0, y0), (x1, y1), HEADING_BORDER_BGR, 2, lineType=cv2.LINE_AA)
                tag = _detector_tag(el, name_counts, seen)
                (tw_h, th_h), _ = cv2.getTextSize(tag, font, 0.5, 1)
                cv2.putText(
                    frame,
                    tag,
                    (max(0, x0), max(th_h + 2, y0 - 3)),
                    font,
                    0.5,
                    HEADING_LABEL_BGR,
                    1,
                    cv2.LINE_AA,
                )
        elif el.name == "paragraph":
            x0 = max(0, min(w - 1, int(round(el.x0))))
            y0 = max(0, min(h - 1, int(round(el.y0))))
            x1 = max(0, min(w - 1, int(round(el.x1))))
            y1 = max(0, min(h - 1, int(round(el.y1))))
            if x1 > x0 + 2 and y1 > y0 + 2:
                cv2.rectangle(frame, (x0, y0), (x1, y1), PARAGRAPH_BORDER_BGR, 1, lineType=cv2.LINE_AA)
                tag = _detector_tag(el, name_counts, seen)
                (tw_p, th_p), _ = cv2.getTextSize(tag, font, 0.4, 1)
                cv2.putText(
                    frame,
                    tag,
                    (max(0, x0), max(th_p + 1, y0 - 2)),
                    font,
                    0.4,
                    PARAGRAPH_LABEL_BGR,
                    1,
                    cv2.LINE_AA,
                )
        elif el.name == "navigation_bar":
            x0 = max(0, min(w - 1, int(round(el.x0))))
            y0 = max(0, min(h - 1, int(round(el.y0))))
            x1 = max(0, min(w - 1, int(round(el.x1))))
            y1 = max(0, min(h - 1, int(round(el.y1))))
            if x1 > x0 + 2 and y1 > y0 + 2:
                cv2.rectangle(frame, (x0, y0), (x1, y1), NAV_ELEMENT_BORDER_BGR, 2, lineType=cv2.LINE_AA)
                tag = _detector_tag(el, name_counts, seen)
                (tw_n, th_n), _ = cv2.getTextSize(tag, font, ts, thk)
                cv2.putText(
                    frame,
                    tag,
                    (max(0, x0), max(th_n + 2, y0 - 4)),
                    font,
                    ts,
                    NAV_ELEMENT_LABEL_BGR,
                    thk,
                    cv2.LINE_AA,
                )

    _draw_detection_legend(frame, elements, w, h)


def _load_final_table(path: str) -> List[Tuple[int, float, Optional[float], Optional[float], str]]:
    rows: List[Tuple[int, float, Optional[float], Optional[float], str]] = []
    with open(path, "r", encoding="utf-8") as f:
        sample = f.readline()
        f.seek(0)
        delim = "\t" if sample.count("\t") >= sample.count(",") else ","
        for row in csv.DictReader(f, delimiter=delim):
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
            el = (row.get("element_name") or row.get("section_name") or "").strip() or "none"
            if has_gaze and has_gaze != "0" and x_s and y_s:
                try:
                    x = float(x_s)
                    y = float(y_s)
                except ValueError:
                    x = y = None
            else:
                x = y = None
            rows.append((frame_idx, ts, x, y, el))
    return rows


def _validation_frame_targets(duration_sec: float) -> List[Tuple[float, str]]:
    """
    Five timestamps spread across the clip (~1s, ~25%, ~50%, ~75%, ~last second).
    Times are nudged apart so short videos still get distinct snapshots.
    """
    d = max(float(duration_sec), 0.5)
    eps = 0.05
    raw: List[Tuple[float, str]] = [
        (min(1.0, max(0.2, d * 0.02)), "frame_1s.png"),
        (d * 0.25, "frame_25pct.png"),
        (d * 0.50, "frame_mid.png"),
        (d * 0.75, "frame_75pct.png"),
        (max(d - 1.0, d * 0.90), "frame_end.png"),
    ]
    out: List[Tuple[float, str]] = []
    for t, name in raw:
        t = max(0.0, min(t, d - eps))
        if out and t <= out[-1][0] + 0.2:
            t = min(out[-1][0] + 0.25, d - eps)
        out.append((t, name))
    return out


def _find_frame_with_gaze(rows, target_time: float, max_delta_sec: float = 1.0):
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


def _load_ppt_rect(output_dir: str) -> Optional[Tuple[float, float, float, float]]:
    path = os.path.join(output_dir, "debug", "ppt_region.txt")
    if not os.path.isfile(path):
        path = os.path.join(output_dir, "ppt_region.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        return None
    parts = lines[1].split("\t")
    if len(parts) < 4:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None


def _inset_rect_tl(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    w: int,
    h: int,
    inset_px: int,
) -> Tuple[int, int, int, int]:
    """Shrink rectangle inward on each side (tighter PPT outline on screen)."""
    xi0 = int(round(x0)) + inset_px
    yi0 = int(round(y0)) + inset_px
    xi1 = int(round(x1)) - inset_px
    yi1 = int(round(y1)) - inset_px
    xi0 = max(0, min(w - 2, xi0))
    yi0 = max(0, min(h - 2, yi0))
    xi1 = max(0, min(w - 1, xi1))
    yi1 = max(0, min(h - 1, yi1))
    if xi1 <= xi0 + 2 or yi1 <= yi0 + 2:
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    return xi0, yi0, xi1, yi1


def _column_is_black_pillar(
    gray: np.ndarray,
    x: int,
    y_top: int,
    y_bot: int,
    dark_thr: int = 40,
    min_frac: float = 0.82,
) -> bool:
    """Column is pillarbox black in the vertical band [y_top, y_bot) (slide-aligned)."""
    col = gray[y_top:y_bot, x]
    if col.size == 0:
        return False
    frac = float((col < dark_thr).mean())
    mean = float(np.mean(col))
    return frac >= min_frac and mean < 48.0


def _detect_side_black_bars_from_slide_edges(
    frame: np.ndarray,
    ppt_rect: Tuple[float, float, float, float],
    y_nav_i: int,
    dark_thr: int = 40,
    min_frac: float = 0.82,
    min_width: int = 3,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Black pillarbox bars sit *beside* the slide inside the embed, not at frame x=0.
    From slide left edge (x0) walk left while columns stay black; from slide right (x1) walk right.
    Vertical extent matches the slide band [y0, y_nav), not the full frame top.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0 = int(round(ppt_rect[0]))
    x1 = int(round(ppt_rect[2]))
    y0 = int(round(ppt_rect[1]))
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y_bot = max(y0 + 8, min(y_nav_i, h))

    def col_black(x: int) -> bool:
        return _column_is_black_pillar(gray, x, y0, y_bot, dark_thr=dark_thr, min_frac=min_frac)

    left_range: Optional[Tuple[int, int]] = None
    if x0 >= 1:
        x = x0 - 1
        while x >= 0 and col_black(x):
            x -= 1
        lx0 = x + 1
        lx1 = x0 - 1
        if lx1 >= lx0 and (lx1 - lx0) >= min_width:
            left_range = (lx0, lx1)

    right_range: Optional[Tuple[int, int]] = None
    if x1 < w - 1:
        x = x1 + 1
        while x < w and col_black(x):
            x += 1
        rx0 = x1 + 1
        rx1 = x - 1
        if rx1 >= rx0 and (rx1 - rx0) >= min_width:
            right_range = (rx0, rx1)

    return left_range, right_range


def _player_x_span(
    left_range: Optional[Tuple[int, int]],
    right_range: Optional[Tuple[int, int]],
    ppt_rect: Tuple[float, float, float, float],
) -> Tuple[int, int]:
    """Horizontal outer bounds of the player (pillars + slide) for red nav overlay."""
    _, _, px1, _ = ppt_rect
    px0 = int(round(ppt_rect[0]))
    px1 = int(round(px1))
    lo = left_range[0] if left_range is not None else px0
    hi = right_range[1] if right_range is not None else px1
    return lo, hi


def _draw_chrome_highlights(
    frame: np.ndarray,
    ppt_rect: Tuple[float, float, float, float],
) -> None:
    """
    Chrome inside the embedded player (not full screenshot):
    - Black pillarbox: columns left of slide x0 / right of slide x1 that read as black
      between ppt y0 and the red nav top.
    - Red nav: bottom strip, horizontally from outer pillar edges (or slide edges).
    """
    h, w = frame.shape[:2]
    px0, py0, px1, _ = [int(round(v)) for v in ppt_rect]
    px0 = max(0, min(w - 1, px0))
    px1 = max(0, min(w - 1, px1))
    py0 = max(0, min(h - 1, py0))

    margin = max(120, int(0.08 * w))
    x_scan_lo = max(0, px0 - margin)
    x_scan_hi = min(w - 1, px1 + margin)

    nav_y = detect_nav_bar_top_y_in_player(frame, x_scan_lo, x_scan_hi)
    nav_y = max(float(NAV_BAR_HEIGHT_PX * 0.35), min(nav_y, float(h - 6)))
    y_nav_i = int(round(nav_y))
    y_nav_i = max(py0 + 1, min(y_nav_i, h - 2))

    narrow_lo, narrow_hi = _player_x_span(None, None, ppt_rect)
    nav_y = detect_nav_bar_top_y_in_player(frame, narrow_lo, narrow_hi)
    nav_y = max(float(NAV_BAR_HEIGHT_PX * 0.35), min(nav_y, float(h - 6)))
    y_nav_i = int(round(nav_y))
    y_nav_i = max(py0 + 1, min(y_nav_i, h - 2))

    y_nav_i = max(py0 + 1, min(y_nav_i, h - 2))

    left_range, right_range = _detect_side_black_bars_from_slide_edges(frame, ppt_rect, y_nav_i)
    if left_range is None or right_range is None:
        lr2, rr2 = _detect_side_black_bars_from_slide_edges(
            frame, ppt_rect, y_nav_i, dark_thr=56, min_frac=0.72, min_width=2
        )
        left_range = left_range or lr2
        right_range = right_range or rr2
    player_lo, player_hi = _player_x_span(left_range, right_range, ppt_rect)

    y_bar_top = py0
    y_bar_bot = max(y_bar_top, min(y_nav_i - 1, h - 1))

    if left_range is not None and y_bar_bot >= y_bar_top:
        lx0, lx1 = left_range
        if lx1 >= lx0:
            cv2.rectangle(
                frame,
                (lx0, y_bar_top),
                (lx1, y_bar_bot),
                BLACK_BAR_OUTLINE_BGR,
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "left black bar",
                (max(4, lx0 + 2), min(h - 4, y_bar_top + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BLACK_BAR_LABEL_BGR,
                1,
                cv2.LINE_AA,
            )

    if right_range is not None and y_bar_bot >= y_bar_top:
        rx0, rx1 = right_range
        if rx1 >= rx0:
            cv2.rectangle(
                frame,
                (rx0, y_bar_top),
                (rx1, y_bar_bot),
                BLACK_BAR_OUTLINE_BGR,
                2,
                lineType=cv2.LINE_AA,
            )
            tw, _ = cv2.getTextSize("right black bar", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(
                frame,
                "right black bar",
                (max(4, rx1 - tw - 4), min(h - 4, y_bar_top + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BLACK_BAR_LABEL_BGR,
                1,
                cv2.LINE_AA,
            )

    y_nav_bot = detect_red_nav_bottom_y(frame, player_lo, player_hi, y_nav_i)
    y_nav_bot = max(y_nav_i, min(h - 1, y_nav_bot))

    if y_nav_bot > y_nav_i and player_hi > player_lo:
        cv2.rectangle(
            frame,
            (player_lo, y_nav_i),
            (player_hi, y_nav_bot),
            NAV_BAR_OUTLINE_BGR,
            2,
            lineType=cv2.LINE_AA,
        )
        nav_txt = "red navigation bar"
        tw, th = cv2.getTextSize(nav_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cx = max(player_lo + 4, min(player_hi - tw - 4, (player_lo + player_hi - tw) // 2))
        ty = min(h - 8, min(y_nav_bot, y_nav_i + th + 8))
        cv2.putText(
            frame,
            nav_txt,
            (cx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            NAV_BAR_LABEL_BGR,
            2,
            cv2.LINE_AA,
        )


def _draw_overlay(
    frame: np.ndarray,
    x_bl: float,
    y_bl: float,
    element_name: str,
    ppt_rect: Optional[Tuple[float, float, float, float]],
    validation_elements: Optional[List[Element]] = None,
    text_frame_fallback: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    h, w = frame.shape[:2]
    if ppt_rect is not None:
        _draw_chrome_highlights(frame, ppt_rect)
        x0, y0, x1, y1 = ppt_rect
        inset = 0
        xi0, yi0, xi1, yi1 = _inset_rect_tl(x0, y0, x1, y1, w, h, inset)
        cv2.rectangle(frame, (xi0, yi0), (xi1, yi1), PPT_BORDER_BGR, 2, lineType=cv2.LINE_AA)
        ppt_tag = "PPT area"
        font_tag = cv2.FONT_HERSHEY_SIMPLEX
        ts, thk = 0.6, 2
        (tw_p, th_p), _ = cv2.getTextSize(ppt_tag, font_tag, ts, thk)
        tx = max(0, min(w - tw_p - 4, xi0))
        ty = max(th_p + 4, yi0 - 6)
        cv2.putText(frame, ppt_tag, (tx, ty), font_tag, ts, PPT_LABEL_BGR, thk, cv2.LINE_AA)
        if validation_elements is not None:
            _draw_validation_detectors(frame, validation_elements, w, h)
        elif text_frame_fallback is not None:
            tx0, ty0, tx1, ty1 = text_frame_fallback
            ax0 = max(0, min(w - 1, int(round(tx0))))
            ay0 = max(0, min(h - 1, int(round(ty0))))
            ax1 = max(0, min(w - 1, int(round(tx1))))
            ay1 = max(0, min(h - 1, int(round(ty1))))
            if ax1 > ax0 + 2 and ay1 > ay0 + 2:
                cv2.rectangle(
                    frame,
                    (ax0, ay0),
                    (ax1, ay1),
                    BUTTON_TEXT_FRAME_BORDER_BGR,
                    2,
                    lineType=cv2.LINE_AA,
                )
                tf_tag = "button_text (no OCR)"
                (tw_t, th_t), _ = cv2.getTextSize(tf_tag, font_tag, 0.5, 1)
                cv2.putText(
                    frame,
                    tf_tag,
                    (max(0, ax0), max(th_t + 4, ay0 - 6)),
                    font_tag,
                    0.5,
                    BUTTON_TEXT_FRAME_LABEL_BGR,
                    1,
                    cv2.LINE_AA,
                )
    x, y = _bl_to_image(x_bl, y_bl, h)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    cv2.circle(frame, (x, y), 7, GAZE_COLOR_BGR, -1)
    cv2.circle(frame, (x, y), 9, (0, 0, 0), 1)
    label = f"looking at: {element_name or 'none'}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 5
    x1 = max(0, x + 10)
    y1 = max(0, y - th - pad * 2)
    x2 = min(w, x1 + tw + pad * 2)
    y2 = min(h, y1 + th + pad * 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + pad, y2 - pad), font, scale, LABEL_TEXT_BGR, thickness, cv2.LINE_AA)


def run_validation(video_path: str, final_csv: str, output_dir: str) -> None:
    rows = _load_final_table(final_csv)
    if not rows:
        print("No rows in", final_csv)
        return
    ppt_rect_fallback = _load_ppt_rect(os.path.dirname(os.path.abspath(final_csv)))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if frame_count > 0 else rows[-1][1]
    targets = _validation_frame_targets(duration_sec)
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    for t_target, name in targets:
        chosen = _find_frame_with_gaze(rows, t_target, max_delta_sec=1.0)
        if chosen is None:
            print(f"No gaze near t≈{t_target:.2f}s; skipping {name}")
            continue
        frame_idx, ts, x_bl, y_bl, sec = chosen
        prev_frame = None
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            rpv, fprev = cap.read()
            if rpv and fprev is not None and fprev.size:
                prev_frame = fprev
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx} for {name}")
            continue
        base, ext = os.path.splitext(name)
        orig_name = base + "_original" + ext
        cv2.imwrite(os.path.join(val_dir, orig_name), frame.copy())
        ppt_rect = detect_ppt_rect_from_chrome(frame)
        if ppt_rect is None:
            ppt_rect = ppt_rect_fallback
            if ppt_rect is not None:
                print(f"  {name}: PPT rect from ppt_region.txt (chrome failed)", flush=True)
        else:
            print(
                f"  {name}: PPT rect from chrome_geometry "
                f"({ppt_rect[0]:.0f},{ppt_rect[1]:.0f})-({ppt_rect[2]:.0f},{ppt_rect[3]:.0f})",
                flush=True,
            )
        validation_elements: Optional[List[Element]] = None
        if ppt_rect is not None:
            det = detect_elements_for_validation(frame, ppt_rect, prev_frame)
            if det is not None:
                validation_elements, _ = det
        text_frame_fallback: Optional[Tuple[float, float, float, float]] = None
        if validation_elements is None and ppt_rect is not None:
            text_frame_fallback = detect_text_frame_popup_rect(frame, ppt_rect, prev_frame)
        _draw_overlay(frame, x_bl, y_bl, sec, ppt_rect, validation_elements, text_frame_fallback)
        cv2.imwrite(os.path.join(val_dir, name), frame)
        print(f"Saved {orig_name}, {name} at t={ts:.2f}s frame={frame_idx} element={sec!r}")
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPT validation: 5 frames with gaze + element + PPT border.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument(
        "--final-csv",
        "--final-table",
        dest="final_csv",
        type=str,
        required=True,
        help="final_gaze_table.txt (tab) or debug/final_gaze_table.csv",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Same folder as final_gaze_table (parent of validation/).")
    args = parser.parse_args()
    run_validation(os.path.abspath(args.video), os.path.abspath(args.final_csv), os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
