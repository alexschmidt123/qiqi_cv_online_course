"""
Step 2 (PPT): Detect slide elements on bucketed frames (default 0.5s), assign element per gaze row.

Gaze is not an element — we only pick which region the viewer looks at (distance in px, top-left).

--- Policy (summary) — full spec: scripts_image/DETECTION_POLICY.md ---

**No PPT:** navigation_bar (nav strip) → blank_area.

**navigation_bar:** narrow **red** strip **between** slide inner edges (same horizontal span as the slide,
red–black at pillars); top edge = slide white → red; bottom = red → white (footer). Not full frame width.

**heading:** **large** **red** title text in the **top band** of the slide only — separate from body **paragraph**
(darker/grey text), which is masked out of paragraph lines via dilated red heading regions.

**Inside PPT:** navigation strip → then, in order: (1) **button_text** — overlay layer on the slide
(pair with **green solid-filled** clicked **button**); popup geometry + neutral gaze patch; (2) **button**
(bordered circles 1–4, i, !; clicked = **green solid fill**); (3)
**chromatic bg** → **heading** or **image**; (4) **image** box; (5) distance / point-in-box /
nearest for **heading**, **paragraph**, etc.; (6) **blank_area**. `_finalize_gaze_label` may remap
slide labels below the effective slide bottom.

**Detectors:** popup = Canny / white / green heuristics; optional **diff vs previous bucket** on PPT
crop → new popup bbox (`debug/ppt_content_change.csv`). Image/heading/paragraph/buttons as in code.

Exported element_name: heading, paragraph, button_text, button, image, navigation_bar, blank_area
(plus none when no gaze).

Usage:
  python scripts_image/step_2_detect_elements.py --video path.mp4 --gaze-csv ... --output-dir ... [--interval 0.5]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import easyocr
except ImportError:
    easyocr = None

from step_90_legacy_article_detect_sections import bl_to_tl  # noqa: E402

from step_80_chrome_geometry import navigation_bar_bbox_tight  # noqa: E402

_SI_DIR = os.path.dirname(os.path.abspath(__file__))
if _SI_DIR not in sys.path:
    sys.path.insert(0, _SI_DIR)
try:
    from step_81_legacy_slide_region_ai import detect_ppt_rect_from_frame_openai
except ImportError:
    detect_ppt_rect_from_frame_openai = None  # type: ignore
try:
    from step_82_legacy_slide_elements_ai import detect_ppt_ui_elements_openai
except ImportError:
    detect_ppt_ui_elements_openai = None  # type: ignore

DEFAULT_INTERVAL_SEC = 0.5
DEFAULT_GAZE_MATCH_THRESHOLD_PX = 18.0
NAV_BAR_HEIGHT_PX = 110
PPT_MARGIN_X_FRAC = 0.02
PPT_MARGIN_TOP_FRAC = 0.04
# Heading = large red title text only in this top fraction of the slide (TL y); not mixed with paragraphs.
HEADING_BAND_FRAC = 0.30
HEADING_MIN_AREA_PX = 120.0
HEADING_MIN_HEIGHT_PX = 16

# Tie-break when distances are equal: matches DETECTION_POLICY.md Step 3 order (heading → … → image).
TIE_PRIORITY = (
    "heading",
    "paragraph",
    "button_1",
    "button_2",
    "button_3",
    "button_4",
    "button_i",
    "button_bang",
    "button_text",
    "image",
    "navigation_bar",
)

# Reject paragraph blobs larger than this fraction of the PPT crop (avoids “whole slide = paragraph”).
PARAGRAPH_MAX_AREA_FRAC = 0.14

# py1 = top of red nav bar (TL y). Small slack only (avoid classifying last slide row as nav).
NAV_TOP_SLACK_PX = 5.0
# If gaze is within this distance (px) of the navigation_bar bbox, label navigation_bar.
NAV_GAZE_NEAR_PX = 10.0
# When py1 and nav_el.y0 disagree, still treat bottom-of-frame as nav (player chrome strip).
NAV_BOTTOM_BAND_EXTRA_PX = 45.0
# If gaze is inside PPT box but no edge within threshold, assign nearest region up to this distance.
GAZE_NEAREST_FALLBACK_PX = 48.0
# Radius (px) around gaze for background color sampling (odd size patch).
GAZE_BG_PATCH_RADIUS = 7
# Mean absdiff on PPT grayscale (0–255) above this ⇒ likely slide interaction (e.g. new popup).
PPT_CONTENT_CHANGE_MEAN_ABS_DIFF_THR = 4.5

# Slide-only labels (never on chrome strip). If stored py1 is loose and includes the nav band, gaze on
# the bar can still be "inside" the PPT rect; remap when y is on/below slide bottom (py1).
_SLIDE_CONTENT_GAZE_LABELS = frozenset(
    {
        "heading",
        "paragraph",
        "button_text",
        "image",
        "button_1",
        "button_2",
        "button_3",
        "button_i",
        "button_bang",
        "button_4",
    }
)

# Exported element_name column (UI types on/near PPT + nav + empty; gaze is not a label).
_BUTTON_INTERNAL = frozenset({"button_1", "button_2", "button_3", "button_4", "button_i", "button_bang"})
@dataclass
class FrameDetectMeta:
    """Per-bucket frame analysis: PPT content change vs previous bucket (inferred interaction)."""

    ppt_diff_mean: float = 0.0
    inferred_content_change: bool = False
    new_text_frame_from_diff: bool = False
    button_text_x0: float = 0.0
    button_text_y0: float = 0.0
    button_text_x1: float = 0.0
    button_text_y1: float = 0.0


def to_public_element_name(internal: str) -> str:
    """Map internal detector labels to public element names (+ none)."""
    n = (internal or "").strip()
    if not n or n.lower() == "none":
        return "none"
    if n in _BUTTON_INTERNAL:
        return "button"
    return n


def _point_in_ppt_rect(x_tl: float, y_tl: float, ppt: Tuple[float, float, float, float]) -> bool:
    px0, py0, px1, py1 = ppt
    return float(px0) <= x_tl <= float(px1) and float(py0) <= y_tl <= float(py1)


def _finalize_gaze_label(
    name: str,
    y_tl: float,
    slide_bottom: float,
    bottom_band_y: float,
) -> str:
    """If gaze is strictly below stored py1, never assign slide UI (handles loose PPT bottom vs chrome)."""
    if name not in _SLIDE_CONTENT_GAZE_LABELS:
        return name
    sb = float(slide_bottom)
    if y_tl <= sb + 1e-6:
        return name
    if y_tl >= bottom_band_y - 5.0 or y_tl >= sb - NAV_GAZE_NEAR_PX:
        return "navigation_bar"
    return "blank_area"


def _extract_gaze_patch(frame_bgr: np.ndarray, x_tl: float, y_tl: float, radius: int) -> np.ndarray:
    """Square patch around gaze (top-left coords), clipped to frame."""
    h, w = frame_bgr.shape[:2]
    xi = int(round(x_tl))
    yi = int(round(y_tl))
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)
    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return frame_bgr[y0:y1, x0:x1].copy()


def _gaze_patch_is_white_or_black_neutral(patch: np.ndarray) -> bool:
    """
    True if the local background reads as slide white/gray, black ink, or dark neutral —
    not a colored figure. If False, policy restricts labels to heading or image inside PPT.
    """
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return True
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    vm = float(np.median(v))
    sm = float(np.median(s))
    if vm >= 192 and sm <= 55:
        return True
    if vm <= 55:
        return True
    if sm <= 45 and 50 <= vm <= 248:
        return True
    return False


def _patch_reads_red_heading(patch: np.ndarray) -> bool:
    """Small patch consistent with maroon/red title text (not diagram-only heuristic)."""
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(patch)
    rm = float(np.median(r.astype(np.float32)))
    gm = float(np.median(g.astype(np.float32)))
    bm = float(np.median(b.astype(np.float32)))
    if (rm > gm + 20) and (rm > bm + 20) and rm > 80:
        return True
    m1 = cv2.inRange(hsv, np.array([0, 38, 42]), np.array([13, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 38, 42]), np.array([180, 255, 255]))
    comb = cv2.bitwise_or(m1, m2)
    if float(np.mean(comb)) > 70.0:
        return True
    return False


def _gaze_pixel_allows_button_text_label(
    frame_bgr: Optional[np.ndarray], x_tl: float, y_tl: float
) -> bool:
    """
    `button_text` = reading text inside the framed popup. If the gaze patch is saturated
    (diagram colors: red wedge, green, blue, …), do not label as button_text even when the
    detector bbox or padding overlaps the diagram — fall through to **image** / bg policy.
    """
    if frame_bgr is None:
        return False
    patch = _extract_gaze_patch(frame_bgr, x_tl, y_tl, GAZE_BG_PATCH_RADIUS)
    if patch.size == 0:
        return True
    return _gaze_patch_is_white_or_black_neutral(patch)


def _resolve_heading_or_image_chromatic(
    x_tl: float,
    y_tl: float,
    ppt: Tuple[float, float, float, float],
    elements: Sequence[Element],
    patch: np.ndarray,
) -> str:
    """Inside PPT, non-white/non-black pixel: must be heading or image."""
    for el in elements:
        if el.name == "heading" and el.x0 <= x_tl <= el.x1 and el.y0 <= y_tl <= el.y1:
            return "heading"
    for el in elements:
        if el.name == "image" and el.x0 <= x_tl <= el.x1 and el.y0 <= y_tl <= el.y1:
            return "image"
    px0, py0, px1, py1 = ppt
    band_y = float(py0) + (float(py1) - float(py0)) * float(HEADING_BAND_FRAC)
    if y_tl <= band_y and _patch_reads_red_heading(patch):
        return "heading"
    return "image"


def _bgr_patch_is_chromatic_colorful(patch: np.ndarray) -> bool:
    """True if patch is not predominantly black, white, or neutral gray (diagram/photo-like)."""
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    vm = float(np.mean(v))
    sm = float(np.mean(s))
    if vm < 40.0:
        return False
    if vm > 235.0 and sm < 30.0:
        return False
    b, g, r = cv2.split(patch.astype(np.int32))
    spread = float(np.mean((np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r))))
    if spread < 22.0 and sm < 28.0:
        return False
    return True


BUTTON_LABELS = {
    "1": "button_1",
    "2": "button_2",
    "3": "button_3",
    "4": "button_4",
    "i": "button_i",
    "I": "button_i",
    "!": "button_bang",
}


@dataclass
class Element:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def distance_to_point(self, x: float, y: float) -> float:
        dx = 0.0
        if x < self.x0:
            dx = self.x0 - x
        elif x > self.x1:
            dx = x - self.x1
        dy = 0.0
        if y < self.y0:
            dy = self.y0 - y
        elif y > self.y1:
            dy = y - self.y1
        return math.hypot(dx, dy)


def _rect_intersection_area_el(a: Element, b: Element) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _button_edge_circularity(frame_bgr: np.ndarray, e: Element) -> float:
    """0..1 from Canny edges in the button bbox; ~1 for round controls, low for wedges/blobs."""
    h, w = frame_bgr.shape[:2]
    xi0 = max(0, int(e.x0))
    yi0 = max(0, int(e.y0))
    xi1 = min(w - 1, int(e.x1))
    yi1 = min(h - 1, int(e.y1))
    if xi1 <= xi0 + 4 or yi1 <= yi0 + 4:
        return 0.5
    crop = frame_bgr[yi0 : yi1 + 1, xi0 : xi1 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 130)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    a = float(cv2.contourArea(c))
    p = float(cv2.arcLength(c, True))
    if p < 1e-6 or a < 25.0:
        return 0.0
    circ = 4.0 * math.pi * a / (p * p)
    return float(min(1.0, circ))


def _filter_spurious_vlm_buttons(
    elements: List[Element],
    frame_bgr: np.ndarray,
    ppt: Tuple[float, float, float, float],
) -> None:
    """
    Remove VLM false positives on buttons:

    - Center inside a heading/paragraph box (e.g. words "the", "test").
    - Large overlap with heading/paragraph.
    - Bbox absurdly large/small vs slide (not a lesson circle).
    - In the diagram band (below paragraph OCR band), require modest edge circularity; pie wedges
      and chart junk score low.
    """
    text_els = [e for e in elements if e.name in ("heading", "paragraph")]
    px0, py0, px1, py1 = (float(ppt[0]), float(ppt[1]), float(ppt[2]), float(ppt[3]))
    slide_area = max(1.0, (px1 - px0) * (py1 - py0))
    # Paragraph detector only uses top ~72% of PPT crop; below = diagram / lesson strip row.
    para_bottom_y = py0 + 0.72 * (py1 - py0)
    # Lesson numbered circles often sit just above the red nav; skip circularity there.
    strip_row_y0 = py0 + 0.78 * (py1 - py0)

    to_remove: List[Element] = []
    for e in elements:
        if e.name not in _BUTTON_INTERNAL:
            continue
        ab = max(1.0, (e.x1 - e.x0) * (e.y1 - e.y0))
        rel = ab / slide_area
        # Carousel pagination dots (~14–16 px) often pass VLM as button_1/2/3; lesson circles are larger.
        if rel > 0.0045 or ab < 400.0:
            to_remove.append(e)
            continue
        cx = (e.x0 + e.x1) * 0.5
        cy = (e.y0 + e.y1) * 0.5

        drop_text = False
        for tx in text_els:
            if tx.x0 <= cx <= tx.x1 and tx.y0 <= cy <= tx.y1:
                drop_text = True
                break
            inter = _rect_intersection_area_el(e, tx)
            if inter / ab >= 0.30:
                drop_text = True
                break
        if drop_text:
            to_remove.append(e)
            continue

        # Diagram / chart junk (colored wedges): middle vertical band, below title row, above
        # paragraph OCR strip — require a roughly circular edge; pie sectors score low.
        title_band_y = py0 + 0.20 * (py1 - py0)
        mid_diagram = title_band_y <= cy < para_bottom_y and cy < strip_row_y0
        if mid_diagram:
            circ = _button_edge_circularity(frame_bgr, e)
            if circ < 0.22:
                to_remove.append(e)

    if not to_remove:
        return
    elements[:] = [e for e in elements if e not in to_remove]


def _tie_rank(name: str) -> int:
    try:
        return TIE_PRIORITY.index(name)
    except ValueError:
        return len(TIE_PRIORITY)


def detect_navigation_bar_rect(frame_h: int, frame_w: int) -> Element:
    y0 = float(frame_h - NAV_BAR_HEIGHT_PX)
    y1 = float(frame_h - 1)
    return Element("navigation_bar", 0.0, y0, float(frame_w - 1), y1)


def detect_ppt_rect(frame_h: int, frame_w: int) -> Tuple[float, float, float, float]:
    x0 = PPT_MARGIN_X_FRAC * frame_w
    x1 = (1.0 - PPT_MARGIN_X_FRAC) * frame_w
    y0 = PPT_MARGIN_TOP_FRAC * frame_h
    y1 = float(frame_h - NAV_BAR_HEIGHT_PX - 4)
    return x0, y0, x1, y1


def detect_nav_bar_top_y(frame: np.ndarray) -> float:
    """
    Top edge (TL y) of the bottom red navigation bar.

    Uses the same logic as chrome_geometry: scan only the lower ~half of the frame so
    the red slide heading is not mistaken for the nav strip.
    """
    from step_80_chrome_geometry import detect_nav_bar_top_y_in_player

    h, w = frame.shape[:2]
    return detect_nav_bar_top_y_in_player(frame, 0, w - 1)


def _detect_ppt_rect_from_frame_column_profile(frame: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Legacy heuristic: middle-band column darkness + nav y (no explicit pillar runs).
    PPT region (top-left coords): first four element types are assigned when gaze falls inside here.
    - x0 / x1: inner vertical limits of side black bars (middle-band column profile).
    - y0: top edge of the top horizontal black bar (use y=0 when letterbox touches frame top).
    - y1: top edge of the red navigation bar.
    """
    h, w = frame.shape[:2]
    nav_y = detect_nav_bar_top_y(frame)
    nav_y = max(float(NAV_BAR_HEIGHT_PX * 0.5), min(nav_y, float(h - 6)))
    y_nav_i = int(nav_y)

    # Middle vertical strip: avoids top/bottom bars so column profile reflects side pillarbox only.
    y_mid0 = int(h * 0.22)
    y_mid1 = max(y_mid0 + 12, y_nav_i - 8)
    if y_mid1 <= y_mid0 + 10:
        y_mid0, y_mid1 = int(h * 0.12), max(int(h * 0.12) + 20, y_nav_i - 5)
    mid = frame[y_mid0:y_mid1, :]
    gray_mid = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    col_dark = (gray_mid < 52).mean(axis=0)

    i = 0
    while i < min(w - 1, w // 2 + 220) and col_dark[i] > 0.32:
        i += 1
    x0 = float(i)

    i = w - 1
    while i > max(0, w // 2 - 220) and col_dark[i] > 0.32:
        i -= 1
    x1 = float(max(i, 0))

    if x1 <= x0 + 80:
        x0, _, x1, _ = detect_ppt_rect(h, w)

    # Top: top *edge* of the top horizontal black bar (not the row below it).
    y_top_end = min(y_nav_i - 4, max(12, int(h * 0.3)))
    strip_top = frame[:y_top_end, :]
    gray_top = cv2.cvtColor(strip_top, cv2.COLOR_BGR2GRAY)
    row_dark_top = (gray_top < 52).mean(axis=1)
    y0 = PPT_MARGIN_TOP_FRAC * h
    if len(row_dark_top) >= 2:
        mean_top = float(np.mean(row_dark_top[: min(8, len(row_dark_top))]))
        if mean_top > 0.2:
            y0 = 0.0
        else:
            for y in range(1, len(row_dark_top)):
                if row_dark_top[y] > 0.34 and row_dark_top[y - 1] <= 0.22:
                    y0 = float(y)
                    break

    y1 = float(nav_y)
    if y1 <= y0 + 40.0:
        _, y0, _, y1 = detect_ppt_rect(h, w)
    return x0, y0, x1, y1


def detect_ppt_rect_from_frame(frame: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Prefer chrome-based PPT box (black pillar runs + red nav top); fall back to
    column-profile heuristic.
    """
    from step_80_chrome_geometry import detect_ppt_rect_from_chrome

    ppt = detect_ppt_rect_from_chrome(frame)
    if ppt is not None:
        return ppt
    return _detect_ppt_rect_from_frame_column_profile(frame)


def _green_clicked_strip_centers(
    frame: np.ndarray, ppt: Tuple[float, float, float, float]
) -> List[Tuple[float, float]]:
    """
    Centers of **green solid-filled** strip controls (post-click). Used to anchor `button_text`
    popups: a frame corner should lie near one of these points.
    """
    px0, py0, px1, py1 = [int(round(v)) for v in ppt]
    h, w = frame.shape[:2]
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(w - 1, px1)
    py1 = min(h - 1, py1)
    ph = py1 - py0
    y_strip0 = int(py0 + ph * 0.72)
    strip = frame[y_strip0 : py1 + 1, px0 : px1 + 1]
    if strip.size == 0:
        return []
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([95, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: List[Tuple[float, float]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 120.0 or a > 14000.0:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1e-6:
            continue
        circ = 4.0 * math.pi * a / (peri * peri)
        if circ < 0.22:
            continue
        m = cv2.moments(c)
        if m["m00"] < 1e-6:
            continue
        cx = float(m["m10"] / m["m00"] + px0)
        cy = float(m["m01"] / m["m00"] + y_strip0)
        centers.append((cx, cy))
    return centers


def _rect_corner_near_green(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    greens: Sequence[Tuple[float, float]],
    max_dist: float,
) -> bool:
    if not greens:
        return False
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    for gx, gy in greens:
        for px, py in corners:
            if math.hypot(px - gx, py - gy) <= max_dist:
                return True
    return False


def _rect_has_dark_border_light_interior(
    frame: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    """Text-frame: closed dark border, bright (paper) interior — typical black frame on white."""
    h, w = frame.shape[:2]
    xi0 = max(0, min(w - 1, int(round(x0))))
    yi0 = max(0, min(h - 1, int(round(y0))))
    xi1 = max(0, min(w - 1, int(round(x1))))
    yi1 = max(0, min(h - 1, int(round(y1))))
    if xi1 <= xi0 + 12 or yi1 <= yi0 + 12:
        return False
    crop = frame[yi0 : yi1 + 1, xi0 : xi1 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape
    band = max(3, min(8, min(ch, cw) // 28))
    peri = np.concatenate(
        [gray[0, :].ravel(), gray[-1, :].ravel(), gray[:, 0].ravel(), gray[:, -1].ravel()]
    )
    if float(np.mean(peri)) > 118.0:
        return False
    inner = gray[band : ch - band, band : cw - band]
    if inner.size == 0:
        return False
    return float(np.mean(inner)) >= 158.0


def _thin_black_border_popup_rect(
    frame: np.ndarray,
    ppt: Tuple[float, float, float, float],
    greens: Sequence[Tuple[float, float]],
) -> Optional[Element]:
    """
    Canny quads inside PPT: keep candidates with dark perimeter + light interior; require a
    **green strip anchor** near a corner when `greens` is non-empty.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in ppt]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w - 1, x1)
    y1 = min(h - 1, y1)
    crop = frame[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 25, 85)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_margin = max(10, min(cw, ch) // 40)
    best: Optional[Tuple[float, float, float, float]] = None
    best_a = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < ch * cw * 0.018 or a > ch * cw * 0.82:
            continue
        bx, by, bw_, bh = cv2.boundingRect(c)
        if bw_ < 52 or bh < 32:
            continue
        if bx <= edge_margin or by <= edge_margin:
            continue
        if bx + bw_ >= cw - edge_margin or by + bh >= ch - edge_margin:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 80:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4 or len(approx) > 14:
            continue
        fx0 = float(x0 + bx)
        fy0 = float(y0 + by)
        fx1 = float(x0 + bx + bw_)
        fy1 = float(y0 + by + bh)
        if not _rect_has_dark_border_light_interior(frame, fx0, fy0, fx1, fy1):
            continue
        if greens:
            if not _rect_corner_near_green(fx0, fy0, fx1, fy1, greens, 150.0):
                continue
        else:
            frac_area = (fx1 - fx0) * (fy1 - fy0) / float(max(1, ch * cw))
            if frac_area < 0.14:
                continue
        if a > best_a:
            best_a = a
            best = (fx0, fy0, fx1, fy1)
    if best is None:
        return None
    return Element("button_text", best[0], best[1], best[2], best[3])


def _white_text_frame_rect(
    frame: np.ndarray,
    ppt: Tuple[float, float, float, float],
    greens: Sequence[Tuple[float, float]],
) -> Optional[Element]:
    """Popup text after a click: bright white/cream frame that does not touch all PPT edges."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in ppt]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w - 1, x1)
    y1 = min(h - 1, y1)
    crop = frame[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 208, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[Tuple[float, float, float, float]] = None
    best_a = 0.0
    edge_margin = max(10, min(cw, ch) // 40)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < ch * cw * 0.035 or a > ch * cw * 0.82:
            continue
        bx, by, bw_, bh = cv2.boundingRect(c)
        if bw_ < 48 or bh < 28:
            continue
        if bx <= edge_margin or by <= edge_margin:
            continue
        if bx + bw_ >= cw - edge_margin or by + bh >= ch - edge_margin:
            continue
        if a > best_a:
            best_a = a
            best = (float(x0 + bx), float(y0 + by), float(x0 + bx + bw_), float(y0 + by + bh))
    if best is None:
        return None
    fx0, fy0, fx1, fy1 = best
    if not greens:
        return None
    if not _rect_corner_near_green(fx0, fy0, fx1, fy1, greens, 150.0):
        return None
    if not _rect_has_dark_border_light_interior(frame, fx0, fy0, fx1, fy1):
        crop = frame[int(fy0) : int(fy1) + 1, int(fx0) : int(fx1) + 1]
        g2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if g2.size == 0 or float(np.mean(g2)) < 182.0:
            return None
    return Element("button_text", fx0, fy0, fx1, fy1)


def _ppt_mean_absdiff_gray(
    prev_bgr: np.ndarray, curr_bgr: np.ndarray, ppt: Tuple[float, float, float, float]
) -> float:
    """Mean absolute grayscale difference on PPT crop (0–255 scale)."""
    h, w = curr_bgr.shape[:2]
    px0, py0, px1, py1 = [int(round(v)) for v in ppt]
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(w - 1, px1)
    py1 = min(h - 1, py1)
    prev_c = prev_bgr[py0 : py1 + 1, px0 : px1 + 1]
    curr_c = curr_bgr[py0 : py1 + 1, px0 : px1 + 1]
    if prev_c.size == 0 or curr_c.size == 0 or prev_c.shape != curr_c.shape:
        return 0.0
    pg = cv2.cvtColor(prev_c, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(curr_c, cv2.COLOR_BGR2GRAY)
    pg = cv2.GaussianBlur(pg, (3, 3), 0)
    cg = cv2.GaussianBlur(cg, (3, 3), 0)
    return float(np.mean(cv2.absdiff(pg, cg)))


def _detect_new_text_frame_from_diff(
    prev_bgr: np.ndarray, curr_bgr: np.ndarray, ppt: Tuple[float, float, float, float]
) -> Optional[Element]:
    """
    Border of UI that *appeared* since the previous frame: largest connected region in
    PPT absdiff mask (new popup / text card after a slide interaction).
    """
    h, w = curr_bgr.shape[:2]
    px0, py0, px1, py1 = [int(round(v)) for v in ppt]
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(w - 1, px1)
    py1 = min(h - 1, py1)
    prev_c = prev_bgr[py0 : py1 + 1, px0 : px1 + 1]
    curr_c = curr_bgr[py0 : py1 + 1, px0 : px1 + 1]
    if prev_c.size == 0 or curr_c.size == 0 or prev_c.shape != curr_c.shape:
        return None
    pg = cv2.cvtColor(prev_c, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(curr_c, cv2.COLOR_BGR2GRAY)
    pg = cv2.GaussianBlur(pg, (3, 3), 0)
    cg = cv2.GaussianBlur(cg, (3, 3), 0)
    diff = cv2.absdiff(pg, cg)
    _, binmask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binmask = cv2.dilate(binmask, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ch, cw = diff.shape[:2]
    crop_area = float(ch * cw)
    best: Optional[Tuple[int, int, int, int]] = None
    best_a = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < crop_area * 0.01 or a > crop_area * 0.85:
            continue
        bx, by, bw_, bh = cv2.boundingRect(c)
        if bw_ < 36 or bh < 28:
            continue
        if a > best_a:
            best_a = a
            best = (bx, by, bw_, bh)
    if best is None:
        return None
    bx, by, bw_, bh = best
    gx0 = float(px0 + bx)
    gy0 = float(py0 + by)
    gx1 = float(px0 + bx + bw_)
    gy1 = float(py0 + by + bh)
    # Tighten box using Canny edges on current crop (popup border).
    xi0, yi0, xi1, yi1 = int(bx), int(by), int(bx + bw_), int(by + bh)
    xi0 = max(0, xi0)
    yi0 = max(0, yi0)
    xi1 = min(curr_c.shape[1], xi1)
    yi1 = min(curr_c.shape[0], yi1)
    roi = curr_c[yi0:yi1, xi0:xi1]
    if roi.size == 0:
        return Element("button_text", gx0, gy0, gx1, gy1)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray_roi, (3, 3), 0), 20, 70)
    ys, xs = np.where(edges > 0)
    if xs.size > 40 and ys.size > 40:
        tx0 = float(px0 + bx + float(np.min(xs)))
        ty0 = float(py0 + by + float(np.min(ys)))
        tx1 = float(px0 + bx + float(np.max(xs)))
        ty1 = float(py0 + by + float(np.max(ys)))
        if tx1 - tx0 > 24 and ty1 - ty0 > 20:
            return Element("button_text", tx0, ty0, tx1, ty1)
    return Element("button_text", gx0, gy0, gx1, gy1)


def detect_text_frame_popup_rect(
    frame: np.ndarray,
    ppt: Tuple[float, float, float, float],
    prev_frame: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Bbox of the text-frame popup — prefers OpenAI scan of PPT crop when configured, else CV heuristics.
    """
    if detect_ppt_ui_elements_openai is not None:
        ui = detect_ppt_ui_elements_openai(frame, ppt)
        if ui:
            pops = [e for e in ui if e.name == "button_text"]
            if pops:
                e = max(pops, key=lambda x: max(0.0, x.x1 - x.x0) * max(0.0, x.y1 - x.y0))
                return (e.x0, e.y0, e.x1, e.y1)

    greens = _green_clicked_strip_centers(frame, ppt)
    popup_el = _thin_black_border_popup_rect(frame, ppt, greens)
    if popup_el is None:
        popup_el = _white_text_frame_rect(frame, ppt, greens)
    if prev_frame is not None and prev_frame.shape == frame.shape:
        if _ppt_mean_absdiff_gray(prev_frame, frame, ppt) >= PPT_CONTENT_CHANGE_MEAN_ABS_DIFF_THR:
            diff_el = _detect_new_text_frame_from_diff(prev_frame, frame, ppt)
            if diff_el is not None:
                dx0, dy0, dx1, dy1 = diff_el.x0, diff_el.y0, diff_el.x1, diff_el.y1
                if _rect_has_dark_border_light_interior(frame, dx0, dy0, dx1, dy1):
                    if not greens or _rect_corner_near_green(dx0, dy0, dx1, dy1, greens, 170.0):
                        popup_el = diff_el
    if popup_el is None:
        return None
    return (popup_el.x0, popup_el.y0, popup_el.x1, popup_el.y1)


_validation_reader: Optional[Any] = None


def get_or_create_validation_reader() -> Optional[Any]:
    """Lazy EasyOCR for step 4 validation overlays (strip buttons + images)."""
    global _validation_reader
    if easyocr is None:
        return None
    if _validation_reader is None:
        _validation_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _validation_reader


def detect_elements_for_validation(
    frame: np.ndarray,
    ppt: Tuple[float, float, float, float],
    prev_frame: Optional[np.ndarray] = None,
    use_openai_ppt_ui: bool = True,
) -> Optional[Tuple[List[Element], FrameDetectMeta]]:
    """Same detectors as the pipeline; returns `(elements, meta)` or None if EasyOCR unavailable."""
    r = get_or_create_validation_reader()
    if r is None:
        return None
    _ppt, elements, _snap, meta = detect_elements_on_frame(
        r, frame, ppt=ppt, prev_frame=prev_frame, use_openai_ppt_ui=use_openai_ppt_ui
    )
    return elements, meta


def _button_name_from_text(txt: str) -> Optional[str]:
    t = re.sub(r"\s+", "", (txt or "").strip())
    if not t:
        return None
    if t in BUTTON_LABELS:
        return BUTTON_LABELS[t]
    low = t.lower()
    if low in ("button_1", "button_2", "button_3", "button_4", "button_i", "button_bang"):
        return low
    return None


def _merge_gpt_button_text_popups(gpt_ui: List[Element]) -> Optional[Element]:
    """If GPT returns multiple `button_text` boxes, keep the largest by area."""
    pops = [e for e in gpt_ui if e.name == "button_text"]
    if not pops:
        return None
    return max(pops, key=lambda e: max(0.0, e.x1 - e.x0) * max(0.0, e.y1 - e.y0))


def _image_regions(
    frame: np.ndarray,
    ppt: Tuple[float, float, float, float],
    text_boxes: List[Tuple[float, float, float, float]],
    strip_button_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Element]:
    """
    Edge-based **chromatic** regions inside PPT (diagrams/figures — not B/W-only).

    `strip_button_boxes`: circular 1/2/3/i/! controls — dilated so numbered circles inside a
    diagram are not labeled as one big `image` without the strip holes.
    """
    h, w = frame.shape[:2]
    px0, py0, px1, py1 = ppt
    xi0, yi0 = int(px0), int(py0)
    xi1, yi1 = int(px1), int(py1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[yi0:yi1, xi0:xi1]
    if crop.size == 0:
        return []
    edges = cv2.Canny(crop, 60, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Element] = []
    strip_pad = 28.0
    sboxes = strip_button_boxes or []

    def _overlap_area(gx0: float, gy0: float, gx1: float, gy1: float, tx0: float, ty0: float, tx1: float, ty1: float) -> float:
        ix0 = max(gx0, tx0)
        iy0 = max(gy0, ty0)
        ix1 = min(gx1, tx1)
        iy1 = min(gy1, ty1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return float((ix1 - ix0) * (iy1 - iy0))

    for c in cnts:
        a = cv2.contourArea(c)
        crop_area = float((xi1 - xi0) * (yi1 - yi0))
        if a < 2500 or a > crop_area * 0.85:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh < 40:
            continue
        gx0 = xi0 + bx
        gy0 = yi0 + by
        gx1 = xi0 + bx + bw
        gy1 = yi0 + by + bh
        overlap = 0.0
        for tx0, ty0, tx1, ty1 in text_boxes:
            overlap += _overlap_area(gx0, gy0, gx1, gy1, tx0, ty0, tx1, ty1)
        for tx0, ty0, tx1, ty1 in sboxes:
            overlap += _overlap_area(
                gx0,
                gy0,
                gx1,
                gy1,
                tx0 - strip_pad,
                ty0 - strip_pad,
                tx1 + strip_pad,
                ty1 + strip_pad,
            )
        if overlap > a * 0.5:
            continue
        y0i, y1i = max(0, int(gy0)), min(h, int(gy1) + 1)
        x0i, x1i = max(0, int(gx0)), min(w, int(gx1) + 1)
        patch = frame[y0i:y1i, x0i:x1i]
        if not _bgr_patch_is_chromatic_colorful(patch):
            continue
        out.append(Element("image", float(gx0), float(gy0), float(gx1), float(gy1)))
    out.sort(key=lambda e: (e.x1 - e.x0) * (e.y1 - e.y0), reverse=True)
    return out[:4]


def _red_heading_elements(
    frame: np.ndarray, ppt: Tuple[float, float, float, float]
) -> Tuple[List[Element], np.ndarray]:
    """
    **Headings:** saturated **red**, **large** glyphs (slide titles) — top band of the slide only.
    Body **paragraph** text (smaller, grey/dark) is detected separately and excluded here; the dilated
    red mask keeps paragraph OCR from snapping to title pixels.
    """
    px0, py0, px1, py1 = [int(round(v)) for v in ppt]
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(frame.shape[1] - 1, px1)
    py1 = min(frame.shape[0] - 1, py1)
    crop = frame[py0 : py1 + 1, px0 : px1 + 1]
    if crop.size == 0:
        return [], np.zeros((0, 0), dtype=np.uint8)
    gh, gw = crop.shape[:2]
    h_hi = max(8, int(gh * HEADING_BAND_FRAC))
    top = crop[:h_hi, :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(top)
    bgr_red = (r.astype(np.int32) > g.astype(np.int32) + 28) & (r.astype(np.int32) > b.astype(np.int32) + 28) & (r > 95)
    m1 = cv2.inRange(hsv, np.array([0, 40, 45]), np.array([11, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 40, 45]), np.array([180, 255, 255]))
    mask_hsv = cv2.bitwise_or(m1, m2)
    mask_bgr = (bgr_red.astype(np.uint8) * 255)
    mask = cv2.bitwise_and(mask_hsv, mask_bgr)
    if cv2.countNonZero(mask) < max(80, int(gw * h_hi * 0.002)):
        mask = mask_hsv
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elements: List[Element] = []
    max_heading_area = float(gw * h_hi) * 0.45
    for c in cnts:
        a = cv2.contourArea(c)
        if a < HEADING_MIN_AREA_PX or a > max_heading_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bh < HEADING_MIN_HEIGHT_PX or bw < 10:
            continue
        if bh > 0 and bw / float(bh) > 42.0:
            continue
        elements.append(
            Element("heading", float(px0 + bx), float(py0 + by), float(px0 + bx + bw), float(py0 + by + bh))
        )
    red_top = cv2.dilate(mask, np.ones((11, 11), np.uint8))
    red_full = np.zeros((gh, gw), dtype=np.uint8)
    red_full[:h_hi, :] = red_top
    red_dilated = cv2.dilate(red_full, np.ones((7, 7), np.uint8))
    return elements, red_dilated


def _paragraph_line_elements(
    frame: np.ndarray,
    ppt: Tuple[float, float, float, float],
    red_dilated: np.ndarray,
    popup_el: Optional[Element],
) -> List[Element]:
    """Dark body text via adaptive threshold; upper ~72% of PPT only (above button row)."""
    px0, py0, px1, py1 = [int(round(v)) for v in ppt]
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(frame.shape[1] - 1, px1)
    py1 = min(frame.shape[0] - 1, py1)
    crop = frame[py0 : py1 + 1, px0 : px1 + 1]
    if crop.size == 0:
        return []
    gh, gw = crop.shape[:2]
    gh_cut = max(1, int(gh * 0.72))
    sub = crop[:gh_cut, :]
    if red_dilated.shape[0] == gh and red_dilated.shape[1] == gw:
        sub_red = red_dilated[:gh_cut, :]
    else:
        sub_red = np.zeros((gh_cut, gw), dtype=np.uint8)
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    bw = cv2.bitwise_and(bw, cv2.bitwise_not(sub_red))
    if popup_el is not None:
        qx0 = max(0, int(popup_el.x0 - px0))
        qy0 = max(0, int(popup_el.y0 - py0))
        qx1 = min(gw, int(popup_el.x1 - px0))
        qy1 = min(gh_cut, int(popup_el.y1 - py0))
        if qx1 > qx0 and qy1 > qy0:
            bw[qy0:qy1, qx0:qx1] = 0
    kw = min(22, max(8, gw // 45))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Element] = []
    crop_area = float(gw * gh_cut)
    max_para = crop_area * PARAGRAPH_MAX_AREA_FRAC
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 180 or a > max_para:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 18 or bh < 8:
            continue
        out.append(
            Element("paragraph", float(px0 + bx), float(py0 + by), float(px0 + bx + bw), float(py0 + by + bh))
        )
    return out


def _button_elements_from_strip_ocr(
    reader: "easyocr.Reader", frame: np.ndarray, ppt: Tuple[float, float, float, float]
) -> List[Tuple[str, float, float, float, float]]:
    """OCR only the bottom strip of the PPT (short control labels)."""
    px0, py0, px1, py1 = ppt
    h = float(py1 - py0)
    y_top = py0 + h * 0.72
    x0i, y0i, x1i, y1i = int(px0), int(y_top), int(px1), int(py1)
    strip = frame[y0i:y1i, x0i:x1i]
    if strip.size == 0:
        return []
    rh, rw = strip.shape[:2]
    if rh < 8 or rw < 20:
        return []
    scale = 1.0
    if rw < 400:
        scale = 400.0 / float(rw)
        strip = cv2.resize(strip, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_CUBIC)
    out: List[Tuple[str, float, float, float, float]] = []
    for bbox, txt, conf in reader.readtext(strip, paragraph=False, detail=1):
        if not txt or (conf or 0) < 0.05:
            continue
        arr = np.array(bbox, dtype=float)
        sx0, sy0 = float(np.min(arr[:, 0])), float(np.min(arr[:, 1]))
        sx1, sy1 = float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))
        sx0, sx1 = sx0 / scale, sx1 / scale
        sy0, sy1 = sy0 / scale, sy1 / scale
        fx0 = float(px0) + sx0
        fx1 = float(px0) + sx1
        fy0 = float(y_top) + sy0
        fy1 = float(y_top) + sy1
        out.append((str(txt).strip(), fx0, fy0, fx1, fy1))
    return out


def detect_elements_on_frame(
    reader: "easyocr.Reader",
    frame: np.ndarray,
    ppt: Optional[Tuple[float, float, float, float]] = None,
    prev_frame: Optional[np.ndarray] = None,
    use_openai_ppt_ui: bool = True,
) -> Tuple[Tuple[float, float, float, float], List[Element], List[str], FrameDetectMeta]:
    meta = FrameDetectMeta()
    h, w = frame.shape[:2]
    if ppt is None:
        ppt = detect_ppt_rect_from_frame(frame)
    px0, py0, px1, py1 = ppt
    nav_top = detect_nav_bar_top_y(frame)
    nav_top = max(0.0, min(nav_top, float(h - 2)))
    nx0, ny0, nx1, ny1 = navigation_bar_bbox_tight(frame, px0, px1, nav_top)
    nav = Element("navigation_bar", nx0, ny0, nx1, ny1)

    gpt_ui: Optional[List[Element]] = None
    if use_openai_ppt_ui and detect_ppt_ui_elements_openai is not None:
        gpt_ui = detect_ppt_ui_elements_openai(frame, ppt)

    use_gpt = gpt_ui is not None and len(gpt_ui) > 0

    popup_el: Optional[Element] = None
    strip_reads: List[Tuple[str, float, float, float, float]] = []

    if use_gpt:
        popup_el = _merge_gpt_button_text_popups(gpt_ui)
        # Do not trust VLM button_text alone: require a real framed popup (dark border + light
        # interior) or a green-solid lesson button near a corner (paired overlay). Otherwise
        # static paragraphs / carousels are mislabeled as button_text (e.g. test2).
        if popup_el is not None:
            greens = _green_clicked_strip_centers(frame, ppt)
            ok_border = _rect_has_dark_border_light_interior(
                frame, popup_el.x0, popup_el.y0, popup_el.x1, popup_el.y1
            )
            ok_green_pair = bool(greens) and _rect_corner_near_green(
                popup_el.x0, popup_el.y0, popup_el.x1, popup_el.y1, greens, 170.0
            )
            if not ok_border and not ok_green_pair:
                popup_el = None
        strip_reads = []
        if prev_frame is not None and prev_frame.shape == frame.shape:
            meta.ppt_diff_mean = _ppt_mean_absdiff_gray(prev_frame, frame, ppt)
            meta.inferred_content_change = meta.ppt_diff_mean >= PPT_CONTENT_CHANGE_MEAN_ABS_DIFF_THR
    else:
        greens = _green_clicked_strip_centers(frame, ppt)
        popup_el = _thin_black_border_popup_rect(frame, ppt, greens)
        if popup_el is None:
            popup_el = _white_text_frame_rect(frame, ppt, greens)

        if prev_frame is not None and prev_frame.shape == frame.shape:
            meta.ppt_diff_mean = _ppt_mean_absdiff_gray(prev_frame, frame, ppt)
            meta.inferred_content_change = meta.ppt_diff_mean >= PPT_CONTENT_CHANGE_MEAN_ABS_DIFF_THR
            if meta.inferred_content_change:
                diff_el = _detect_new_text_frame_from_diff(prev_frame, frame, ppt)
                if diff_el is not None:
                    dx0, dy0, dx1, dy1 = diff_el.x0, diff_el.y0, diff_el.x1, diff_el.y1
                    if _rect_has_dark_border_light_interior(frame, dx0, dy0, dx1, dy1):
                        if not greens or _rect_corner_near_green(dx0, dy0, dx1, dy1, greens, 170.0):
                            popup_el = diff_el
                            meta.new_text_frame_from_diff = True
        strip_reads = _button_elements_from_strip_ocr(reader, frame, ppt)

    headings, red_dilated = _red_heading_elements(frame, ppt)
    paragraphs = _paragraph_line_elements(frame, ppt, red_dilated, popup_el)

    elements: List[Element] = []
    if popup_el is not None:
        elements.append(popup_el)
    elements.extend(headings)
    elements.extend(paragraphs)
    if use_gpt:
        for e in gpt_ui:
            if e.name in _BUTTON_INTERNAL:
                elements.append(e)
    else:
        for txt, x0, y0, x1, y1 in strip_reads:
            bn = _button_name_from_text(txt)
            if bn:
                elements.append(Element(bn, x0, y0, x1, y1))

    _filter_spurious_vlm_buttons(elements, frame, ppt)

    text_boxes: List[Tuple[float, float, float, float]] = []
    overlap_boxes_for_image: List[Tuple[float, float, float, float]] = []
    for e in elements:
        if e.name in (
            "heading",
            "paragraph",
            "button_1",
            "button_2",
            "button_3",
            "button_i",
            "button_bang",
            "button_text",
        ):
            t = (e.x0, e.y0, e.x1, e.y1)
            text_boxes.append(t)
            if e.name != "button_text":
                overlap_boxes_for_image.append(t)

    strip_boxes = [(e.x0, e.y0, e.x1, e.y1) for e in elements if e.name in _BUTTON_INTERNAL]
    imgs = _image_regions(frame, ppt, overlap_boxes_for_image, strip_button_boxes=strip_boxes)
    elements.extend(imgs)
    elements.append(nav)
    snapshot_lines = [t for t, _, _, _, _ in strip_reads if t]
    if popup_el is not None:
        meta.button_text_x0 = float(popup_el.x0)
        meta.button_text_y0 = float(popup_el.y0)
        meta.button_text_x1 = float(popup_el.x1)
        meta.button_text_y1 = float(popup_el.y1)
    return ppt, elements, snapshot_lines, meta


@dataclass
class _GazePolicyCtx:
    """Shared state for gaze→label policy (see DETECTION_POLICY.md section C)."""

    x_tl: float
    y_tl: float
    ppt: Tuple[float, float, float, float]
    elements: Sequence[Element]
    frame_h: int
    threshold_px: float
    frame_bgr: Optional[np.ndarray]
    slide_bottom: float
    bottom_band_y: float
    nav_strip_top: float
    nav_el: Optional[Element]

    def out(self, name: str) -> str:
        return _finalize_gaze_label(name, self.y_tl, self.slide_bottom, self.bottom_band_y)

    def nav_strip_matches(self) -> bool:
        if self.y_tl >= self.slide_bottom - NAV_GAZE_NEAR_PX:
            return True
        if self.y_tl >= self.bottom_band_y:
            return True
        if self.nav_el is not None and self.nav_el.distance_to_point(self.x_tl, self.y_tl) <= NAV_GAZE_NEAR_PX:
            return True
        if self.y_tl >= self.slide_bottom - NAV_TOP_SLACK_PX:
            return True
        if self.y_tl >= float(self.frame_h) - float(NAV_BAR_HEIGHT_PX) - 20.0:
            return True
        return False


def _policy_A_no_ppt(c: _GazePolicyCtx) -> Optional[str]:
    """A2: outside PPT → navigation_bar or blank_area only."""
    if _point_in_ppt_rect(c.x_tl, c.y_tl, c.ppt):
        return None
    return "navigation_bar" if c.nav_strip_matches() else "blank_area"


def _policy_A3_nav_inside_ppt(c: _GazePolicyCtx) -> Optional[str]:
    """A3: inside PPT but gaze on nav strip (loose rect) → navigation_bar."""
    if not c.nav_strip_matches():
        return None
    return c.out("navigation_bar")


def _policy_C1_button_text(c: _GazePolicyCtx) -> Optional[str]:
    """C1: `button_text` only if a **detected** popup box exists AND gaze is **inside** it (and neutral patch).

    No distance/padding fallback: if no overlay was detected in this bucket, there is no
    `button_text` element in `c.elements` and policy never assigns `button_text` here.
    """
    bt_el = next((e for e in c.elements if e.name == "button_text"), None)
    if bt_el is None:
        return None
    inside = bt_el.x0 <= c.x_tl <= bt_el.x1 and bt_el.y0 <= c.y_tl <= bt_el.y1
    if not inside:
        return None
    if c.y_tl >= c.bottom_band_y:
        return c.out("navigation_bar")
    if not _gaze_pixel_allows_button_text_label(c.frame_bgr, c.x_tl, c.y_tl):
        return None
    return c.out("button_text")


def _policy_C2_strip_buttons_distance(c: _GazePolicyCtx) -> Optional[str]:
    """C2a: strip circles — nearest within threshold."""
    best_btn: Optional[Tuple[float, str]] = None
    for el in c.elements:
        if el.name not in _BUTTON_INTERNAL:
            continue
        d = el.distance_to_point(c.x_tl, c.y_tl)
        if d > c.threshold_px:
            continue
        if best_btn is None or d < best_btn[0] - 1e-6:
            best_btn = (d, el.name)
        elif math.isclose(d, best_btn[0]) and _tie_rank(el.name) < _tie_rank(best_btn[1]):
            best_btn = (d, el.name)
    if best_btn is not None:
        return c.out(best_btn[1])
    return None


def _policy_C2_strip_buttons_point_in(c: _GazePolicyCtx) -> Optional[str]:
    """C2b: strip circles — gaze inside button bbox."""
    for el in c.elements:
        if el.name not in _BUTTON_INTERNAL:
            continue
        if el.x0 <= c.x_tl <= el.x1 and el.y0 <= c.y_tl <= el.y1:
            return c.out(el.name)
    return None


def _policy_C3_chromatic_heading_or_image(c: _GazePolicyCtx) -> Optional[str]:
    """C3: saturated local patch → heading or image only."""
    if c.frame_bgr is None:
        return None
    # Recording gaze overlay (red dot) on body text makes the patch look saturated; prefer
    # paragraph/heading via C5/C6 when the point is already inside a text box.
    for el in c.elements:
        if el.name not in ("paragraph", "heading"):
            continue
        if el.x0 <= c.x_tl <= el.x1 and el.y0 <= c.y_tl <= el.y1:
            return None
    bg_patch = _extract_gaze_patch(c.frame_bgr, c.x_tl, c.y_tl, GAZE_BG_PATCH_RADIUS)
    if bg_patch.size == 0 or _gaze_patch_is_white_or_black_neutral(bg_patch):
        return None
    return c.out(
        _resolve_heading_or_image_chromatic(float(c.x_tl), float(c.y_tl), c.ppt, c.elements, bg_patch)
    )


def _policy_C4_image_box(c: _GazePolicyCtx) -> Optional[str]:
    """C4: gaze inside detector `image` rectangle."""
    for el in c.elements:
        if el.name == "image" and el.x0 <= c.x_tl <= el.x1 and el.y0 <= c.y_tl <= el.y1:
            return c.out("image")
    return None


def _policy_C5_distance_heading_paragraph(c: _GazePolicyCtx) -> Optional[str]:
    """C5: nearest heading/paragraph/… within threshold (excludes image, button_text, strip)."""
    best: Optional[Tuple[float, str]] = None
    for el in c.elements:
        if el.name in ("navigation_bar", "image", "button_text") or el.name in _BUTTON_INTERNAL:
            continue
        d = el.distance_to_point(c.x_tl, c.y_tl)
        if d > c.threshold_px:
            continue
        if best is None:
            best = (d, el.name)
        elif d < best[0] - 1e-6:
            best = (d, el.name)
        elif math.isclose(d, best[0]) and _tie_rank(el.name) < _tie_rank(best[1]):
            best = (d, el.name)
    if best is not None:
        return c.out(best[1])
    return None


def _policy_C6_point_in_remaining(c: _GazePolicyCtx) -> Optional[str]:
    """C6: point inside any remaining element box (not button_text / strip)."""
    for el in c.elements:
        if el.name in ("navigation_bar", "button_text") or el.name in _BUTTON_INTERNAL:
            continue
        if el.x0 <= c.x_tl <= el.x1 and el.y0 <= c.y_tl <= el.y1:
            return c.out(el.name)
    return None


def _policy_C7_nearest_fallback(c: _GazePolicyCtx) -> Optional[str]:
    """C7: nearest element within GAZE_NEAREST_FALLBACK_PX on slide."""
    if c.y_tl > c.slide_bottom + 1e-6:
        return None
    nearest: Optional[Tuple[float, str]] = None
    for el in c.elements:
        if el.name in ("navigation_bar", "image", "button_text") or el.name in _BUTTON_INTERNAL:
            continue
        d = el.distance_to_point(c.x_tl, c.y_tl)
        if nearest is None or d < nearest[0]:
            nearest = (d, el.name)
    if nearest is not None and nearest[0] <= GAZE_NEAREST_FALLBACK_PX:
        return c.out(nearest[1])
    return None


def choose_element_for_gaze(
    x_tl: float,
    y_tl: float,
    ppt: Tuple[float, float, float, float],
    elements: Sequence[Element],
    frame_h: int,
    threshold_px: float,
    frame_bgr: Optional[np.ndarray] = None,
) -> str:
    """
    Apply gaze→element policy in **fixed order** (DETECTION_POLICY.md): A2 → A3 → C1 → C2 → C3 →
    C4 → C5 → C6 → C7 → C8 (`blank_area`). C1 only assigns ``button_text`` when gaze is **inside**
    a detected ``button_text`` box and the gaze patch is neutral (no near-padding path).
    """
    px0, py0, px1, py1 = ppt
    nav_el = next((e for e in elements if e.name == "navigation_bar"), None)
    slide_bottom = float(py1)
    if nav_el is not None:
        slide_bottom = min(slide_bottom, float(nav_el.y0))
    bottom_band_y = float(frame_h) - float(NAV_BAR_HEIGHT_PX) - NAV_BOTTOM_BAND_EXTRA_PX
    nav_strip_top = slide_bottom

    c = _GazePolicyCtx(
        x_tl=x_tl,
        y_tl=y_tl,
        ppt=ppt,
        elements=elements,
        frame_h=frame_h,
        threshold_px=threshold_px,
        frame_bgr=frame_bgr,
        slide_bottom=slide_bottom,
        bottom_band_y=bottom_band_y,
        nav_strip_top=nav_strip_top,
        nav_el=nav_el,
    )

    for step in (
        _policy_A_no_ppt,
        _policy_A3_nav_inside_ppt,
        _policy_C1_button_text,
        _policy_C2_strip_buttons_distance,
        _policy_C2_strip_buttons_point_in,
        _policy_C3_chromatic_heading_or_image,
        _policy_C4_image_box,
        _policy_C5_distance_heading_paragraph,
        _policy_C6_point_in_remaining,
        _policy_C7_nearest_fallback,
    ):
        label = step(c)
        if label is not None:
            return label
    return c.out("blank_area")


def _write_final_gaze_txt(csv_path: str, txt_path: str) -> None:
    """Mirror final_gaze_table.csv as tab-separated final_gaze_table.txt for delivery."""
    if not os.path.isfile(csv_path):
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "") or "") for c in cols) + "\n")


def _load_gaze(path: str) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                idx = int(row.get("frame_idx", 0))
            except (TypeError, ValueError):
                idx = 0
            try:
                ts = float(row.get("timestamp_sec", 0))
            except (TypeError, ValueError):
                ts = 0.0
            xs = (row.get("x_bl") or "").strip()
            ys = (row.get("y_bl") or "").strip()
            xb = float(xs) if xs else None
            yb = float(ys) if ys else None
            rows.append((idx, ts, xb, yb))
    return rows


def process_video(
    video_path: str,
    gaze_csv: str,
    output_dir: str,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    threshold_px: float = DEFAULT_GAZE_MATCH_THRESHOLD_PX,
    ppt_cv_only: bool = False,
    openai_ppt_ui: bool = True,
) -> Dict[str, Any]:
    if easyocr is None:
        raise RuntimeError("easyocr is required for PPT element detection.")

    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    gaze_rows = _load_gaze(gaze_csv)
    if not gaze_rows:
        raise RuntimeError(f"No gaze rows in {gaze_csv}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ret0, frame0 = cap.read()
    if ret0 and frame0 is not None:
        h_video, w_video = frame0.shape[:2]
    else:
        h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    bucket_to_frame_ts: Dict[int, Tuple[int, float]] = {}
    for frame_idx, ts, xb, yb in gaze_rows:
        if xb is None or yb is None:
            continue
        b = int(ts / interval_sec)
        if b not in bucket_to_frame_ts:
            bucket_to_frame_ts[b] = (frame_idx, ts)

    sorted_buckets = sorted(bucket_to_frame_ts.keys())
    n_buckets = len(sorted_buckets)

    unified_ppt: Optional[Tuple[float, float, float, float]] = None
    ppt_method = "cv"
    if sorted_buckets:
        b0 = sorted_buckets[0]
        fidx0, _ts0 = bucket_to_frame_ts[b0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx0)
        ret0, frame_ref = cap.read()
        if ret0 and frame_ref is not None:
            from step_80_chrome_geometry import detect_ppt_rect_from_chrome

            unified_ppt = detect_ppt_rect_from_chrome(frame_ref)
            if unified_ppt is not None:
                ppt_method = "chrome"
                print(
                    f"  PPT region: chrome (pillars + red nav) -> "
                    f"({unified_ppt[0]:.0f},{unified_ppt[1]:.0f})-({unified_ppt[2]:.0f},{unified_ppt[3]:.0f})",
                    flush=True,
                )
            elif not ppt_cv_only and detect_ppt_rect_from_frame_openai is not None:
                print("  PPT region: OpenAI vision on first sampled bucket frame...", flush=True)
                unified_ppt = detect_ppt_rect_from_frame_openai(frame_ref)
                if unified_ppt is not None:
                    ppt_method = "openai"
                    print(f"  PPT region: OpenAI OK -> ({unified_ppt[0]:.0f},{unified_ppt[1]:.0f})-({unified_ppt[2]:.0f},{unified_ppt[3]:.0f})", flush=True)
                else:
                    print("  PPT region: OpenAI failed or unavailable; using column-profile CV.", flush=True)
            if unified_ppt is None:
                unified_ppt = _detect_ppt_rect_from_frame_column_profile(frame_ref)
                ppt_method = "cv_first_bucket"

    if unified_ppt is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret0, frame_ref = cap.read()
        if ret0 and frame_ref is not None:
            from step_80_chrome_geometry import detect_ppt_rect_from_chrome

            unified_ppt = detect_ppt_rect_from_chrome(frame_ref)
            if unified_ppt is not None:
                ppt_method = "chrome_frame0"
            else:
                unified_ppt = _detect_ppt_rect_from_frame_column_profile(frame_ref)
                ppt_method = "cv_frame0"
        if unified_ppt is None:
            cap.release()
            raise RuntimeError("Could not determine PPT rectangle (no reference frame).")

    method_path = os.path.join(debug_dir, "ppt_region_method.txt")
    with open(method_path, "w", encoding="utf-8") as mf:
        mf.write(ppt_method + "\n")

    print(
        f"  PPT elements: {n_buckets} buckets @ {interval_sec}s — "
        f"fixed PPT rect ({ppt_method}); "
        f"labels: heading, paragraph, button_text, button, image, navigation_bar, blank_area.",
        flush=True,
    )
    print("  PPT elements: loading EasyOCR (fallback strip OCR when OpenAI UI off)...", flush=True)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    if openai_ppt_ui and detect_ppt_ui_elements_openai is not None:
        print(
            "  PPT elements: OpenAI vision (VLM) scans **PPT slide crop** each bucket for bordered "
            "circle buttons + button_text (set --no-openai-ppt-ui for CV+OCR only).",
            flush=True,
        )
    print(f"  PPT elements: processing {n_buckets} buckets...", flush=True)

    ppt_per_bucket: Dict[int, Tuple[float, float, float, float]] = {}
    elements_per_bucket: Dict[int, List[Element]] = {}
    ocr_snapshot_text: Optional[str] = None
    prev_bucket_frame: Optional[np.ndarray] = None
    content_change_rows: List[List[Any]] = []

    for bi, b in enumerate(sorted_buckets):
        fidx, ts = bucket_to_frame_ts[b]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        h, w = frame.shape[:2]
        ppt, els, snap_lines, meta = detect_elements_on_frame(
            reader,
            frame,
            ppt=unified_ppt,
            prev_frame=prev_bucket_frame,
            use_openai_ppt_ui=openai_ppt_ui,
        )
        ppt_per_bucket[b] = ppt
        elements_per_bucket[b] = els
        content_change_rows.append(
            [
                b,
                fidx,
                f"{ts:.4f}",
                f"{meta.ppt_diff_mean:.4f}",
                int(meta.inferred_content_change),
                int(meta.new_text_frame_from_diff),
                f"{meta.button_text_x0:.1f}",
                f"{meta.button_text_y0:.1f}",
                f"{meta.button_text_x1:.1f}",
                f"{meta.button_text_y1:.1f}",
            ]
        )
        prev_bucket_frame = frame.copy()
        if ocr_snapshot_text is None and snap_lines:
            ocr_snapshot_text = "\n".join(snap_lines[:400])
        if bi == 0 or (bi + 1) % 25 == 0 or bi + 1 == n_buckets:
            print(
                f"  PPT elements: bucket {bi + 1}/{n_buckets} (t≈{ts:.1f}s)...",
                flush=True,
            )

    cap.release()

    change_csv = os.path.join(debug_dir, "ppt_content_change.csv")
    with open(change_csv, "w", newline="", encoding="utf-8") as cf:
        cw = csv.writer(cf)
        cw.writerow(
            [
                "bucket",
                "frame_idx",
                "timestamp_sec",
                "ppt_mean_absdiff_vs_prev",
                "inferred_content_change",
                "new_text_frame_from_diff",
                "button_text_x0",
                "button_text_y0",
                "button_text_x1",
                "button_text_y1",
            ]
        )
        cw.writerows(content_change_rows)
    print(f"  PPT interaction log: {change_csv}", flush=True)

    for b in sorted_buckets:
        if b in ppt_per_bucket:
            ppt_per_bucket[b] = unified_ppt

    debug_csv = os.path.join(debug_dir, "step2_element_buckets.csv")
    with open(debug_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bucket",
                "frame_idx",
                "timestamp_sec",
                "ppt_x0",
                "ppt_y0",
                "ppt_x1",
                "ppt_y1",
                "element_names",
            ]
        )
        for b in sorted_buckets:
            if b not in ppt_per_bucket:
                continue
            fidx, ts = bucket_to_frame_ts[b]
            px = ppt_per_bucket[b]
            names = ";".join(sorted({e.name for e in elements_per_bucket.get(b, [])}))
            w.writerow([b, fidx, ts, px[0], px[1], px[2], px[3], names])

    n_frames = len(gaze_rows)
    print(f"  PPT elements: assigning element labels to {n_frames} gaze rows...", flush=True)
    cap_labels = cv2.VideoCapture(video_path)
    labels_cap_ok = cap_labels.isOpened()
    last_bucket_for_frame: Optional[int] = None
    frame_for_gaze: Optional[np.ndarray] = None

    final_rows: List[Tuple[Any, ...]] = []
    for j, (frame_idx, ts, x_bl, y_bl) in enumerate(gaze_rows):
        if x_bl is None or y_bl is None:
            final_rows.append((frame_idx, ts, "0", "", "", "none"))
            continue
        x_tl, y_tl = bl_to_tl(x_bl, y_bl, h_video)
        if x_tl is None or y_tl is None:
            final_rows.append((frame_idx, ts, "1", str(x_bl), str(y_bl), "none"))
            continue
        b = int(ts / interval_sec)
        if labels_cap_ok and b in bucket_to_frame_ts:
            if last_bucket_for_frame != b:
                fidx_b, _ = bucket_to_frame_ts[b]
                cap_labels.set(cv2.CAP_PROP_POS_FRAMES, fidx_b)
                ret_l, fr = cap_labels.read()
                frame_for_gaze = fr if ret_l and fr is not None else None
                last_bucket_for_frame = b
        else:
            frame_for_gaze = None
            last_bucket_for_frame = None

        ppt = ppt_per_bucket.get(b) or unified_ppt
        els = elements_per_bucket.get(b, [])
        if ppt is None:
            el_name = "blank_area"
        else:
            el_name = to_public_element_name(
                choose_element_for_gaze(
                    float(x_tl),
                    float(y_tl),
                    ppt,
                    els,
                    int(h_video),
                    threshold_px,
                    frame_bgr=frame_for_gaze,
                )
            )
        final_rows.append((frame_idx, ts, "1", str(x_bl), str(y_bl), el_name))
        if (j + 1) % 2000 == 0 or j + 1 == n_frames:
            print(f"  PPT elements: labeled {j + 1}/{n_frames} frames...", flush=True)

    cap_labels.release()

    gaze_element_csv = os.path.join(debug_dir, "gaze_with_element.csv")
    with open(gaze_element_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_sec", "x_bl", "y_bl", "element_name"])
        for frame_idx, ts, hg, xb, yb, eln in final_rows:
            if hg == "0":
                w.writerow([frame_idx, ts, "", "", "none"])
            else:
                w.writerow([frame_idx, ts, xb, yb, eln])

    final_path = os.path.join(debug_dir, "final_gaze_table.csv")
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_sec", "has_gaze", "gaze_x", "gaze_y", "element_name"])
        for frame_idx, ts, hg, gx, gy, eln in final_rows:
            w.writerow([frame_idx, ts, hg, gx, gy, eln])

    final_txt = os.path.join(output_dir, "final_gaze_table.txt")
    _write_final_gaze_txt(final_path, final_txt)

    ppt_debug = os.path.join(debug_dir, "ppt_region.txt")
    if unified_ppt is not None:
        with open(ppt_debug, "w", encoding="utf-8") as pf:
            pf.write(
                "ppt_x0_tl\tppt_y0_tl\tppt_x1_tl\tppt_y1_tl\n"
                f"{unified_ppt[0]}\t{unified_ppt[1]}\t{unified_ppt[2]}\t{unified_ppt[3]}\n"
            )

    snap_path_out: Optional[str] = None
    if ocr_snapshot_text:
        snap_path_out = os.path.join(debug_dir, "ppt_ocr_snapshot.txt")
        with open(snap_path_out, "w", encoding="utf-8") as sf:
            sf.write(ocr_snapshot_text.strip() + "\n")

    return {
        "gaze_with_element_csv": gaze_element_csv,
        "final_gaze_table": final_path,
        "final_gaze_table_txt": final_txt,
        "debug_element_buckets": debug_csv,
        "ppt_content_change_csv": change_csv,
        "ppt_region": unified_ppt,
        "ppt_ocr_snapshot": snap_path_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PPT: detect elements and assign gaze element names.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--gaze-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_GAZE_MATCH_THRESHOLD_PX,
        help="Gaze-to-element distance (px).",
    )
    parser.add_argument(
        "--ppt-cv-only",
        action="store_true",
        help="Do not call OpenAI for PPT rectangle; use CV on first bucket frame only.",
    )
    parser.add_argument(
        "--no-openai-ppt-ui",
        action="store_true",
        help="Do not call OpenAI vision on the PPT crop for buttons/button_text; use CV + strip OCR.",
    )
    args = parser.parse_args()
    stats = process_video(
        os.path.abspath(args.video),
        os.path.abspath(args.gaze_csv),
        os.path.abspath(args.output_dir),
        interval_sec=args.interval,
        threshold_px=args.threshold,
        ppt_cv_only=args.ppt_cv_only,
        openai_ppt_ui=not args.no_openai_ppt_ui,
    )
    print("  Wrote", stats["final_gaze_table"], "and", stats.get("final_gaze_table_txt", ""))


if __name__ == "__main__":
    main()
