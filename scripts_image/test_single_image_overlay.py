"""
Single PNG test: run the same detection + validation-style overlay as step 4 (one frame).

Writes only two files per image:
  - original.png   — copy of the input
  - labeled.png    — chrome + PPT + detectors + gaze dot + "looking at: …"

Single image: output_image/<image_stem>/
Folder batch:  output_image/test_result/<image_stem>/  (each input file)

Usage:
  python scripts_image/test_single_image_overlay.py --image path/to/capture.png
  python scripts_image/test_single_image_overlay.py --folder data/images
"""
from __future__ import annotations

import argparse
import os
import sys
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

_SI_DIR = os.path.dirname(os.path.abspath(__file__))
if _SI_DIR not in sys.path:
    sys.path.insert(0, _SI_DIR)
_ROOT = os.path.dirname(_SI_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from script.step_2_detect_section import bl_to_tl  # noqa: E402

from step_2_detect_elements import (  # noqa: E402
    DEFAULT_GAZE_MATCH_THRESHOLD_PX,
    choose_element_for_gaze,
    detect_elements_for_validation,
    detect_ppt_rect_from_frame,
    detect_text_frame_popup_rect,
    to_public_element_name,
)
from step_4_validate_image import _draw_overlay  # noqa: E402

from chrome_geometry import detect_ppt_rect_from_chrome  # noqa: E402


def _tl_to_bl(x_tl: float, y_tl: float, frame_h: int) -> Tuple[float, float]:
    return float(x_tl), float((frame_h - 1) - y_tl)


def _default_gaze_bl_ppt_center(ppt: Tuple[float, float, float, float], frame_h: int) -> Tuple[float, float]:
    x0, y0, x1, y1 = ppt
    cx_tl = (float(x0) + float(x1)) * 0.5
    cy_tl = (float(y0) + float(y1)) * 0.5
    return _tl_to_bl(cx_tl, cy_tl, frame_h)


def detect_gaze_overlay_center_bl(
    frame: np.ndarray,
    ppt: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float]]:
    """
    Locate a **small red** gaze marker (recording overlay) inside the PPT crop.

    Typical appearance: **solid red disk** with a **slightly darker red ring** border. The border
    is darker (lower value) than the fill; both sit in red hue. We use a wide-enough HSV band to
    catch fill + ring, then **morphological closing** so the ring and interior merge into one blob
    for a stable centroid and circularity.

    Ignores large regions (nav strip, wide UI) and large lesson circles by tight area / size bounds
    so numbered buttons are not mistaken for the gaze dot.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Include darker-red border (lower V) and slightly lower S; widen hue slightly.
    m1 = cv2.inRange(hsv, (0, 28, 20), (15, 255, 255))
    m2 = cv2.inRange(hsv, (165, 28, 20), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    # Bridge ring outline to solid fill so the contour is one disk (not a thin ring / split).
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if ppt is not None:
        px0, py0, px1, py1 = [int(round(v)) for v in ppt]
        px0 = max(0, min(w - 1, px0))
        py0 = max(0, min(h - 1, py0))
        px1 = max(0, min(w - 1, px1))
        py1 = max(0, min(h - 1, py1))
        inset = 4
        ax0 = min(px0 + inset, w - 1)
        ay0 = min(py0 + inset, h - 1)
        ax1 = max(ax0 + 1, px1 - inset)
        ay1 = max(ay0 + 1, py1 - inset)
        region = np.zeros_like(mask)
        region[ay0 : ay1 + 1, ax0 : ax1 + 1] = mask[ay0 : ay1 + 1, ax0 : ax1 + 1]
        mask = region

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Typical gaze: ~10–22 px diameter → area ~80–380. Sorting by circularity alone favors tiny
    # square artifacts (circularity ≈ π/4). Prefer area near a solid disk + darker ring (~140 px²),
    # then roundness as tie-breaker.
    target_area = 140.0
    best: Optional[Tuple[float, float, float, float]] = None  # -|a-target|, circ, cx, cy

    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < 55.0 or a > 520.0:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1e-6:
            continue
        circ = 4.0 * math.pi * a / (peri * peri)
        if circ < 0.58:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) > 32 or min(bw, bh) < 5:
            continue
        if bw > 70:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        key = (-abs(a - target_area), circ)
        if best is None or key > (best[0], best[1]):
            best = (-abs(a - target_area), circ, cx, cy)

    if best is None:
        return None
    _d, _circ, cx, cy = best
    return _tl_to_bl(cx, cy, h)


def _list_image_files(folder: str) -> List[str]:
    """Sorted list of .png/.jpg/.jpeg paths (non-recursive)."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    exts = {".png", ".jpg", ".jpeg", ".jpeg"}
    names: List[str] = []
    for n in sorted(os.listdir(folder)):
        low = n.lower()
        if any(low.endswith(e) for e in (".png", ".jpg", ".jpeg")):
            names.append(os.path.join(folder, n))
    return names


def run_single_image(
    image_path: str,
    output_root: str,
    gaze_x_bl: Optional[float],
    gaze_y_bl: Optional[float],
    threshold_px: float,
    use_openai_ppt_ui: bool,
    out_dir: Optional[str] = None,
    verbose: bool = True,
    detect_gaze_overlay: bool = True,
) -> str:
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    low = image_path.lower()
    if not (low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg")):
        raise ValueError("Expected .png or .jpg image")

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = frame.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.abspath(output_root), stem)
    else:
        out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    orig_out = os.path.join(out_dir, "original.png")
    cv2.imwrite(orig_out, frame)

    ppt = detect_ppt_rect_from_chrome(frame)
    if ppt is None:
        ppt = detect_ppt_rect_from_frame(frame)
        if ppt is not None and verbose:
            print("  PPT rect: column-profile fallback (chrome failed)", flush=True)
    elif verbose:
        print(
            f"  PPT rect: chrome ({ppt[0]:.0f},{ppt[1]:.0f})-({ppt[2]:.0f},{ppt[3]:.0f})",
            flush=True,
        )

    if gaze_x_bl is not None and gaze_y_bl is not None:
        x_bl, y_bl = float(gaze_x_bl), float(gaze_y_bl)
    elif detect_gaze_overlay and ppt is not None:
        guessed = detect_gaze_overlay_center_bl(frame, ppt)
        if guessed is not None:
            x_bl, y_bl = guessed
            if verbose:
                print(
                    f"  Gaze (red overlay in PPT crop, BL): x_bl={x_bl:.1f}, y_bl={y_bl:.1f}",
                    flush=True,
                )
        else:
            x_bl, y_bl = _default_gaze_bl_ppt_center(ppt, h)
            if verbose:
                print(f"  Gaze (default: PPT center BL): x_bl={x_bl:.1f}, y_bl={y_bl:.1f}", flush=True)
    elif ppt is not None:
        x_bl, y_bl = _default_gaze_bl_ppt_center(ppt, h)
        if verbose:
            print(f"  Gaze (default: PPT center BL): x_bl={x_bl:.1f}, y_bl={y_bl:.1f}", flush=True)
    else:
        x_bl, y_bl = _tl_to_bl(w * 0.5, h * 0.5, h)
        if verbose:
            print(f"  Gaze (default: frame center BL): x_bl={x_bl:.1f}, y_bl={y_bl:.1f}", flush=True)

    x_tl, y_tl = bl_to_tl(x_bl, y_bl, h)
    if x_tl is None or y_tl is None:
        x_tl, y_tl = float(w // 2), float(h // 2)

    validation_elements = None
    text_frame_fallback = None
    if ppt is not None:
        det = detect_elements_for_validation(
            frame, ppt, None, use_openai_ppt_ui=use_openai_ppt_ui
        )
        if det is not None:
            validation_elements, _ = det
        if validation_elements is None:
            text_frame_fallback = detect_text_frame_popup_rect(frame, ppt, None)

    els = list(validation_elements) if validation_elements is not None else []

    el_internal = "none"
    if ppt is not None and els:
        el_internal = choose_element_for_gaze(
            float(x_tl),
            float(y_tl),
            ppt,
            els,
            int(h),
            float(threshold_px),
            frame_bgr=frame,
        )
    element_public = to_public_element_name(el_internal)

    overlay = frame.copy()
    _draw_overlay(
        overlay,
        x_bl,
        y_bl,
        element_public,
        ppt,
        validation_elements,
        text_frame_fallback,
    )
    labeled_out = os.path.join(out_dir, "labeled.png")
    cv2.imwrite(labeled_out, overlay)

    if verbose:
        print(f"  Wrote: {orig_out}", flush=True)
        print(f"  Wrote: {labeled_out} (looking at: {element_public})", flush=True)
    return out_dir


def run_folder(
    folder: str,
    output_root: str,
    gaze_x_bl: Optional[float],
    gaze_y_bl: Optional[float],
    threshold_px: float,
    use_openai_ppt_ui: bool,
    test_result_subdir: str = "test_result",
    detect_gaze_overlay: bool = True,
) -> str:
    """
    Run ``run_single_image`` for each image in ``folder`` (non-recursive).
    Subfolders: ``<output_root>/<test_result_subdir>/<stem>/`` with ``original.png`` + ``labeled.png``.
    """
    paths = _list_image_files(folder)
    if not paths:
        raise FileNotFoundError(f"No .png/.jpg images in: {folder}")
    base = os.path.join(os.path.abspath(output_root), test_result_subdir)
    os.makedirs(base, exist_ok=True)
    print(f"=== Batch: {len(paths)} image(s) -> {base}/", flush=True)
    for i, p in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        out_sub = os.path.join(base, stem)
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}", flush=True)
        run_single_image(
            p,
            output_root,
            gaze_x_bl,
            gaze_y_bl,
            threshold_px,
            use_openai_ppt_ui,
            out_dir=out_sub,
            verbose=False,
            detect_gaze_overlay=detect_gaze_overlay,
        )
        print(f"    -> {out_sub}/ (original.png, labeled.png)", flush=True)
    print(f"Done. {len(paths)} folder(s) under {base}/", flush=True)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test PPT element detectors on one PNG or a folder; write original + labeled overlay only."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Single .png or .jpg (screenshot).")
    src.add_argument("--folder", type=str, help="Directory of images (non-recursive); outputs under test_result/.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Parent folder (default: project root output_image). Single image: <root>/<stem>/; batch: <root>/test_result/<stem>/",
    )
    parser.add_argument(
        "--test-result-subdir",
        type=str,
        default="test_result",
        help="With --folder only: subdirectory under output-root (default: test_result).",
    )
    parser.add_argument(
        "--gaze-x-bl",
        type=float,
        default=None,
        help="Gaze x in bottom-left coordinates (default: PPT center).",
    )
    parser.add_argument(
        "--gaze-y-bl",
        type=float,
        default=None,
        help="Gaze y in bottom-left coordinates (default: PPT center).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_GAZE_MATCH_THRESHOLD_PX,
        help=f"Gaze→element distance threshold (default {DEFAULT_GAZE_MATCH_THRESHOLD_PX}).",
    )
    parser.add_argument(
        "--no-openai-ppt-ui",
        action="store_true",
        help="Skip OpenAI vision for buttons/button_text (CV + strip OCR only).",
    )
    parser.add_argument(
        "--no-detect-gaze-overlay",
        action="store_true",
        help="Do not infer gaze from a small red marker in the PPT area; use PPT center (unless --gaze-* set).",
    )
    args = parser.parse_args()

    out_root = args.output_root
    if out_root is None:
        out_root = os.path.join(_ROOT, "output_image")

    gx, gy = args.gaze_x_bl, args.gaze_y_bl
    if (gx is None) ^ (gy is None):
        parser.error("Provide both --gaze-x-bl and --gaze-y-bl, or neither for default PPT-center gaze.")

    use_oa = not args.no_openai_ppt_ui

    detect_overlay = not args.no_detect_gaze_overlay
    if args.folder:
        run_folder(
            os.path.abspath(args.folder),
            out_root,
            gx,
            gy,
            float(args.threshold),
            use_oa,
            test_result_subdir=args.test_result_subdir.strip() or "test_result",
            detect_gaze_overlay=detect_overlay,
        )
    else:
        run_single_image(
            args.image,
            out_root,
            gx,
            gy,
            float(args.threshold),
            use_oa,
            detect_gaze_overlay=detect_overlay,
        )


if __name__ == "__main__":
    main()
