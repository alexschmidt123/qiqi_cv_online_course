"""Legacy slide-player geometry helpers.
PPT / slide rectangle from **chrome geometry**:

- Left/right **black pillarbox** columns (dark vertical runs beside the slide).
- Bottom **red** navigation bar → **y1** = top of that strip (slide white → red). The bar’s
  **left/right** edges align with **red → black** pillar borders (same width as slide content); its
  **bottom** is **red → white** (footer). We detect red only in the **lower** ~half of the frame so
  the slide’s red **heading/title** is not mistaken for this strip.
- **Slide (PPT)** = inner rectangle between the inner edges of the pillars, from
  slide top **y0** down to **y1**.

This does not depend on fixed 0–1000 coordinates; it detects pillars + red per frame.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

NAV_BAR_HEIGHT_PX = 110


def _hsv_red_mask(hsv: np.ndarray) -> np.ndarray:
    m1 = cv2.inRange(hsv, np.array([0, 35, 45]), np.array([15, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 35, 45]), np.array([180, 255, 255]))
    return cv2.bitwise_or(m1, m2)


def detect_red_nav_bottom_y(
    frame: np.ndarray,
    x_lo: int,
    x_hi: int,
    y_nav_top: int,
) -> int:
    """
    Bottom edge (TL y, inclusive) of the red navigation strip inside [x_lo, x_hi].

    Scans upward from the frame bottom so rows below the strip (white footer, copyright)
    are skipped; returns the last row that still has strong red in the player band.
    """
    h, w = frame.shape[:2]
    x_lo = max(0, min(w - 1, int(x_lo)))
    x_hi = max(0, min(w - 1, int(x_hi)))
    if x_hi < x_lo:
        x_lo, x_hi = x_hi, x_lo
    y_nav_top = max(0, min(y_nav_top, h - 2))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = _hsv_red_mask(hsv)
    y = h - 1
    while y > y_nav_top:
        row = red[y, x_lo : x_hi + 1]
        if row.size == 0:
            y -= 1
            continue
        frac = float(row.mean()) / 255.0
        if frac > 0.06:
            return y
        y -= 1
    return min(h - 1, y_nav_top + max(8, NAV_BAR_HEIGHT_PX // 3))


def detect_nav_bar_top_y_in_player(
    frame: np.ndarray,
    x_lo: int,
    x_hi: int,
) -> float:
    """
    Top edge (TL y) of the bottom red navigation strip inside [x_lo, x_hi].

    The slide heading/title is often red; it is the main other red element. We only
    scan rows **below** ~52% of frame height so the heading is excluded. In that band,
    the wide red control strip is typically the only strong horizontal red region.
    """
    h, w = frame.shape[:2]
    x_lo = max(0, min(w - 1, int(x_lo)))
    x_hi = max(0, min(w - 1, int(x_hi)))
    if x_hi < x_lo:
        x_lo, x_hi = x_hi, x_lo

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = _hsv_red_mask(hsv)
    roi = red[:, x_lo : x_hi + 1]
    # Exclude upper slide (red heading, maroon text); nav lives in the lower ~48% of frame
    y_floor = int(h * 0.52)
    # Ensure a minimum band height on short frames
    y_floor = min(y_floor, max(0, h - max(96, NAV_BAR_HEIGHT_PX + 24)))
    band = roi[y_floor:h, :]
    if band.size == 0:
        return float(h - NAV_BAR_HEIGHT_PX)

    row_frac = band.mean(axis=1) / 255.0
    # Full-width red bar: high row density; pagination dots stay below strong threshold
    strong = 0.16
    for i in range(len(row_frac)):
        if row_frac[i] >= strong:
            nav_top = y_floor + i
            return float(max(8, min(nav_top, h - 6)))

    for i in range(len(row_frac)):
        if row_frac[i] >= 0.08:
            nav_top = y_floor + i
            return float(max(8, min(nav_top, h - 6)))
    return float(h - NAV_BAR_HEIGHT_PX)


def _dark_runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    """True runs as inclusive (a, b) indices."""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(mask_1d)
    while i < n:
        if not mask_1d[i]:
            i += 1
            continue
        j = i
        while j < n and mask_1d[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _best_pillar_pair(
    col_dark_frac: np.ndarray,
    w: int,
    thr: float,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Left pillar = rightmost dark run in the left half (next to slide).
    Right pillar = leftmost dark run in the right half.
    """
    is_d = col_dark_frac > thr
    runs = _dark_runs(is_d)
    left_cand = [
        r
        for r in runs
        if r[1] < int(w * 0.46) and 5 <= (r[1] - r[0] + 1) <= 220
    ]
    right_cand = [
        r
        for r in runs
        if r[0] > int(w * 0.54) and 5 <= (r[1] - r[0] + 1) <= 220
    ]
    if not left_cand or not right_cand:
        return None
    L = max(left_cand, key=lambda r: r[1])
    R = min(right_cand, key=lambda r: r[0])
    if L[1] >= R[0] - 8:
        return None
    return L, R


def _slide_top_y(
    frame: np.ndarray,
    x0: int,
    x1: int,
    y_nav_i: int,
    margin_top_frac: float = 0.04,
) -> float:
    """Fallback: top edge of slide from center strip (letterbox above slide)."""
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    if x1 <= x0 + 10:
        return float(margin_top_frac * h)
    y_top_end = min(y_nav_i - 4, max(12, int(h * 0.3)))
    strip_top = frame[:y_top_end, x0 : x1 + 1]
    gray_top = cv2.cvtColor(strip_top, cv2.COLOR_BGR2GRAY)
    row_dark_top = (gray_top < 52).mean(axis=1)
    y0 = margin_top_frac * h
    if len(row_dark_top) >= 2:
        mean_top = float(np.mean(row_dark_top[: min(8, len(row_dark_top))]))
        if mean_top > 0.2:
            y0 = 0.0
        else:
            for y in range(1, len(row_dark_top)):
                if row_dark_top[y] > 0.34 and row_dark_top[y - 1] <= 0.22:
                    y0 = float(y)
                    break
    return float(y0)


def _refine_slide_x_between_pillars(
    gray: np.ndarray,
    h: int,
    w: int,
    lx0: int,
    lx1: int,
    rx0: int,
    rx1: int,
    y0: float,
    y_nav_i: int,
) -> Tuple[float, float]:
    """
    Slide left/right edges = inner vertical borders of the side black bars (no extra padding).

    Uses a vertical band between slide top and nav top; finds the first column after the left
    pillar and the last column before the right pillar where darkness drops vs pillar means.
    """
    y_top = int(max(0, min(h - 2, round(y0) + 4)))
    y_bot = int(max(y_top + 12, min(h - 1, y_nav_i - 4)))
    if y_bot <= y_top + 8:
        return float(lx1 + 1), float(rx0 - 1)
    col_dark = (gray[y_top:y_bot, :] < 42).mean(axis=0)
    lx0 = max(0, min(w - 1, lx0))
    lx1 = max(lx0, min(w - 1, lx1))
    rx0 = max(0, min(w - 1, rx0))
    rx1 = max(rx0, min(w - 1, rx1))
    left_mean = float(np.mean(col_dark[lx0 : lx1 + 1])) if lx1 >= lx0 else 0.85
    right_mean = float(np.mean(col_dark[rx0 : rx1 + 1])) if rx1 >= rx0 else 0.85
    thr_l = max(0.22, min(0.62, left_mean * 0.58))
    thr_r = max(0.22, min(0.62, right_mean * 0.58))
    x0 = float(lx1 + 1)
    for x in range(lx1 + 1, min(rx0, w)):
        if col_dark[x] < thr_l:
            x0 = float(x)
            break
    x1 = float(rx0 - 1)
    for x in range(rx0 - 1, max(int(x0) - 1, lx1), -1):
        if col_dark[x] < thr_r:
            x1 = float(x)
            break
    if x1 <= x0 + 40.0:
        return float(lx1 + 1), float(rx0 - 1)
    return x0, x1


def _slide_top_y_from_pillar_tops(
    gray: np.ndarray,
    h: int,
    left_range: Tuple[int, int],
    right_range: Tuple[int, int],
    y_nav_i: int,
) -> Optional[float]:
    """
    Top y of slide = same horizontal line as the top of the side black pillarbox bars.
    Finds the first row (from below browser chrome) where each pillar column run is dark.
    """
    dark_thr = 42
    y_start = int(0.05 * h)
    y_lim = min(y_nav_i, h - 1)

    def top_for_pillar(xa: int, xb: int) -> Optional[int]:
        for y in range(y_start, y_lim):
            patch = gray[y, xa : xb + 1]
            if patch.size == 0:
                continue
            if float((patch < dark_thr).mean()) < 0.70:
                continue
            y2 = min(y + 14, y_lim)
            sub = gray[y:y2, xa : xb + 1]
            if sub.size == 0:
                continue
            if float((sub < dark_thr).mean()) > 0.55:
                return y
        return None

    lt = top_for_pillar(left_range[0], left_range[1])
    rt = top_for_pillar(right_range[0], right_range[1])
    if lt is None and rt is None:
        return None
    if lt is None:
        return float(rt)
    if rt is None:
        return float(lt)
    return float(min(lt, rt))


def detect_ppt_rect_from_chrome(frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (x0,y0,x1,y1) slide area in pixels, or None if pillars/nav cannot be found.

    x0 = column after left black bar; x1 = column before right black bar;
    y1 = top of red nav (bottom of slide).
    """
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]

    # Pillar column profile: exclude bottom strip (red nav + margin) so we do not need y_nav first.
    y_b0 = int(h * 0.14)
    y_b1 = max(y_b0 + 24, h - max(130, NAV_BAR_HEIGHT_PX + 20))
    if y_b1 <= y_b0 + 8:
        return None

    gray = cv2.cvtColor(frame[y_b0:y_b1, :], cv2.COLOR_BGR2GRAY)
    col_dark_frac = (gray < 42).mean(axis=0)
    if w >= 9:
        k = np.ones(9, dtype=np.float64) / 9.0
        col_dark_frac = np.convolve(col_dark_frac, k, mode="same")

    pair: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    for thr in (0.72, 0.65, 0.58, 0.50, 0.45):
        pair = _best_pillar_pair(col_dark_frac, w, thr)
        if pair is not None:
            break
    if pair is None:
        return None

    (lx0, lx1), (rx0, rx1) = pair
    x0 = float(lx1 + 1)
    x1 = float(rx0 - 1)
    if x1 <= x0 + 80.0:
        return None

    # Nav top: red HSV only across the player (outer pillar edges), not full frame (avoids wrong y).
    y_nav = detect_nav_bar_top_y_in_player(
        frame,
        max(0, lx0 - 10),
        min(w - 1, rx1 + 10),
    )
    # Bottom of slide = top edge of red nav only (no artificial lift toward frame top).
    y_nav_i = int(round(min(max(y_nav, 8.0), float(h - 6))))
    y1 = float(y_nav_i)

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    y0_opt = _slide_top_y_from_pillar_tops(gray_full, h, (lx0, lx1), (rx0, rx1), y_nav_i)
    xi0, xi1 = int(round(x0)), int(round(x1))
    if y0_opt is not None:
        y0 = y0_opt
    else:
        y0 = float(_slide_top_y(frame, xi0, xi1, y_nav_i))

    x0, x1 = _refine_slide_x_between_pillars(
        gray_full, h, w, lx0, lx1, rx0, rx1, y0, y_nav_i
    )

    if y1 <= y0 + 40.0:
        return None
    return (x0, y0, x1, y1)


def navigation_bar_bbox_tight(
    frame: np.ndarray,
    slide_x0: float,
    slide_x1: float,
    nav_top_y: float,
) -> Tuple[float, float, float, float]:
    """
    Bottom **red** navigation strip as a **narrow** bar aligned with the slide content:

    - **Horizontal:** same inner span as the slide (between left/right **black** pillar edges), i.e.
      red meets black at the sides — **not** full browser width.
    - **Top:** ``nav_top_y`` = transition from slide (white/content) **into** red.
    - **Bottom:** last row of strong red in that band (transition **red → white** below, e.g. footer).

    If the slide x-range is degenerate, falls back to full frame width.
    """
    h, w = frame.shape[:2]
    xi0 = int(round(max(0, min(w - 1, slide_x0))))
    xi1 = int(round(max(0, min(w - 1, slide_x1))))
    if xi1 <= xi0 + 4:
        xi0, xi1 = 0, w - 1
    yt = int(round(max(0, min(h - 2, nav_top_y))))
    yb = detect_red_nav_bottom_y(frame, xi0, xi1, yt)
    yb = max(yt + 6, min(h - 1, yb))
    return float(xi0), float(yt), float(xi1), float(yb)
