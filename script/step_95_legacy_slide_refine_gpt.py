"""
Step 3 (optional): Use ChatGPT to refine PPT gaze → element_name labels (bucket-level QA + correction).

Input (in output_image/<video_name>/debug/):
  - gaze_with_element.csv   (from step_2_detect_elements.py)
  - ppt_ocr_snapshot.txt    (optional; button-strip OCR snippet)

Output:
  - debug/final_gaze_table.csv (overwritten)
  - final_gaze_table.txt       (tab-separated; same rows, for delivery)
  - debug/gaze_element_gpt_report.md
  - debug/step3_element_buckets.csv

Usage (from project root):
  python scripts_image/step_3_refine_elements_gpt.py --output-dir output_image/VIDEO_NAME [--interval 0.5]

Requires:
  - OPENAI_API_KEY in environment or .env at project root
  - openai, python-dotenv (see requirements.txt)
"""
from __future__ import annotations

import argparse
import csv
import difflib
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SI_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SI_DIR not in sys.path:
    sys.path.insert(0, _SI_DIR)

from step_90_legacy_article_detect_sections import bl_to_tl  # noqa: E402
from step_94_legacy_slide_detect_elements import (  # noqa: E402
    NAV_BAR_HEIGHT_PX,
    NAV_BOTTOM_BAND_EXTRA_PX,
    NAV_GAZE_NEAR_PX,
    NAV_TOP_SLACK_PX,
)

DEFAULT_INTERVAL_SEC = 0.5

# Public element labels (slide UI + nav + empty; gaze is not a label)
ALLOWED_ELEMENTS: Set[str] = {
    "heading",
    "paragraph",
    "button_text",
    "button",
    "image",
    "navigation_bar",
    "blank_area",
    "none",
}

# Slide content labels must not apply when gaze is outside the PPT content box (tight rect; same as step 4).
_SLIDE_LABELS_FOR_CLAMP: Set[str] = {"heading", "paragraph", "button_text", "button", "image"}


def _load_ppt_rect_from_output_dir(output_dir: str) -> Optional[Tuple[float, float, float, float]]:
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
    """Optional per-side shrink (step 4 uses inset 0 for tight chrome-aligned PPT)."""
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


def _gaze_inside_ppt_inset(
    x_tl: float,
    y_tl: float,
    ppt: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> bool:
    x0, y0, x1, y1 = ppt
    inset = 0
    xi0, yi0, xi1, yi1 = _inset_rect_tl(x0, y0, x1, y1, frame_w, frame_h, inset)
    return float(xi0) <= x_tl <= float(xi1) and float(yi0) <= y_tl <= float(yi1)


def _nav_strip_matches_for_clamp(y_tl: float, py1: float, frame_h: int) -> bool:
    """Bottom-strip heuristic aligned with step 2 (no per-frame nav bbox)."""
    slide_bottom = float(py1)
    bottom_band_y = float(frame_h) - float(NAV_BAR_HEIGHT_PX) - float(NAV_BOTTOM_BAND_EXTRA_PX)
    if y_tl >= slide_bottom - NAV_GAZE_NEAR_PX:
        return True
    if y_tl >= bottom_band_y:
        return True
    if y_tl >= slide_bottom - NAV_TOP_SLACK_PX:
        return True
    if y_tl >= float(frame_h) - float(NAV_BAR_HEIGHT_PX) - 20.0:
        return True
    return False


def _clamp_gpt_label_to_outside_ppt_policy(
    x_tl: float,
    y_tl: float,
    ppt: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    element_name: str,
) -> str:
    """
    If gaze is outside the inset PPT box (matches validation blue border), GPT may not assign
    slide labels — only navigation_bar or blank_area (nav strip wins).
    """
    if element_name not in _SLIDE_LABELS_FOR_CLAMP:
        return element_name
    if _gaze_inside_ppt_inset(x_tl, y_tl, ppt, frame_w, frame_h):
        return element_name
    if _nav_strip_matches_for_clamp(y_tl, float(ppt[3]), frame_h):
        return "navigation_bar"
    return "blank_area"


def _resolve_video_path(output_dir: str, video_arg: Optional[str]) -> Optional[str]:
    if video_arg and os.path.isfile(video_arg):
        return os.path.abspath(video_arg)
    base = os.path.basename(os.path.normpath(output_dir))
    candidate = os.path.join(_PROJECT_ROOT, "data", "video_image", f"{base}.mp4")
    if os.path.isfile(candidate):
        return candidate
    return None


def _video_frame_size(video_path: str) -> Optional[Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap.release()
    if w <= 0 or h <= 0:
        return None
    return w, h


def _get_openai_client() -> OpenAI:
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or to .env in the project root."
        )
    return OpenAI(api_key=api_key)


def _load_gaze_with_element(path: str) -> Tuple[Counter, int, List[Dict[str, str]]]:
    counts: Counter = Counter()
    total = 0
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
            total += 1
            el = (row.get("element_name") or "").strip()
            if el and el.lower() != "none":
                counts[el] += 1
    return counts, total, rows


def _build_element_per_bucket(rows: List[Dict[str, str]], interval_sec: float) -> Dict[int, str]:
    """Majority vote per time bucket (same idea as script/step_3 section buckets)."""
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

    bucket_to_elems: Dict[int, List[str]] = defaultdict(list)
    for row in rows:
        try:
            ts = float(row.get("timestamp_sec", 0))
        except (TypeError, ValueError):
            continue
        bucket = int(ts / interval_sec)
        el = (row.get("element_name") or "").strip()
        bucket_to_elems[bucket].append(el if el else "none")

    if not bucket_to_elems:
        return {}

    min_bucket = min(bucket_to_elems.keys())
    max_bucket = max(bucket_to_elems.keys())

    bucket_primary: Dict[int, str] = {}
    for b, elems in bucket_to_elems.items():
        n = len(elems)
        n_none = sum(1 for e in elems if not e or e.lower() == "none")
        non_none = [e for e in elems if e and e.lower() != "none"]
        if not non_none:
            bucket_primary[b] = "none"
        elif n_none > n / 2:
            bucket_primary[b] = "none"
        else:
            bucket_primary[b] = Counter(non_none).most_common(1)[0][0]

    filled: Dict[int, str] = {}
    last_el = ""
    for b in range(min_bucket, max_bucket + 1):
        if b not in bucket_primary:
            filled[b] = last_el
            continue
        if bucket_primary[b] == "none":
            filled[b] = "none"
            last_el = ""
        else:
            filled[b] = bucket_primary[b]
            last_el = bucket_primary[b]

    first_non_none = next(
        (b for b in range(min_bucket, max_bucket + 1) if filled.get(b) and filled[b] != "none"),
        None,
    )
    if first_non_none is not None:
        el0 = filled[first_non_none]
        for b in range(min_bucket, first_non_none):
            if not filled.get(b):
                filled[b] = el0

    return filled


def _build_prompt_ppt(
    video_name: str,
    ocr_snapshot: str,
    element_counts: Counter,
    total_rows: int,
) -> str:
    top_el = element_counts.most_common()
    counts_str = "\n".join(f"- {name}: {count} gaze frames" for name, count in top_el) or "(no elements)"
    ocr_block = (ocr_snapshot or "").strip()[:12000]
    return f"""
You are reviewing a screen recording of a PowerPoint-style slide deck. The viewer's gaze is shown as a small red circle.

VIDEO: {video_name}

PART 1 — OCR SNAPSHOT (first sampled frame; may be incomplete)
---
{ocr_block}
---

PART 2 — GAZE→ELEMENT COUNTS
Automated pipeline assigned each frame with gaze to one element (nearest UI region within ~10 px).
Total rows with gaze: {total_rows}. Counts per element label:

{counts_str}

Allowed element names are exactly (gaze is not an element — it is only used to infer which region is viewed):
heading, paragraph, button_text, button, image, navigation_bar, blank_area, none

Respond with:
1) A short section "PPT ELEMENT QA" with 2–4 bullets: whether the count distribution looks plausible for a slide deck, and any obvious mislabels the automation might make.
2) A section "RECOMMENDATIONS" with 1–2 bullets on rules to improve labeling (high level only).
"""


def _build_refine_elements_prompt(
    cleaned_qa: str,
    element_per_bucket: Dict[int, str],
    interval_sec: float,
) -> str:
    if not element_per_bucket:
        return ""
    buckets = sorted(element_per_bucket.keys())
    lines = [
        f"Bucket {b} ({b * interval_sec:.2f}s–{(b + 1) * interval_sec:.2f}s): {element_per_bucket[b]}"
        for b in buckets
    ]
    return f"""
You refine gaze→element labels for a PPT video. Time buckets are {interval_sec:.2f}s each.

Rules:
- Use **button_text** only when a **modal/overlay text frame** on the slide is plausibly present for that segment (not plain slide titles or body copy alone).
- Use **button** only for **small circular lesson controls** with a single digit (1–4) or **i** / **!** inside.

Context from reviewer:
---
{cleaned_qa[:6000]}
---

PRELIMINARY ELEMENT PER BUCKET:
{chr(10).join(lines)}

For each bucket, output EXACTLY ONE line with the element name only (one of:
heading, paragraph, button_text, button, image, navigation_bar, blank_area, none).
Line 1 = bucket {buckets[0]}, line 2 = bucket {buckets[1]}, etc. No numbering, no extra text.
"""


def _parse_refined_elements(response: str, buckets: List[int]) -> Dict[int, str]:
    refined: Dict[int, str] = {}
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    for i, b in enumerate(buckets):
        if i >= len(lines):
            break
        name = lines[i].strip()
        # Strip leading "bucket N:" or "1." if model adds them
        for prefix in ("bucket",):
            if name.lower().startswith(prefix):
                parts = name.split(":", 1)
                if len(parts) > 1:
                    name = parts[1].strip()
        refined[b] = name
    return refined


def _write_final_gaze_txt(csv_path: str, txt_path: str) -> None:
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


def _load_step2_bucket_element_names(debug_dir: str) -> Dict[int, str]:
    """Bucket index -> semicolon-separated detector names from step 2 (see step2_element_buckets.csv)."""
    path = os.path.join(debug_dir, "step2_element_buckets.csv")
    out: Dict[int, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                b = int(row.get("bucket", 0))
            except (TypeError, ValueError):
                continue
            out[b] = (row.get("element_names") or "").strip()
    return out


def _step2_bucket_has_button_text_overlay(names_str: str) -> bool:
    parts = [p.strip() for p in (names_str or "").split(";") if p.strip()]
    return "button_text" in parts


def _clamp_button_text_without_detection(
    element_name: str,
    bucket: int,
    step2_bucket_names: Dict[int, str],
    pre_el: str,
) -> str:
    """Do not keep ``button_text`` if step 2 did not detect a ``button_text`` overlay in that bucket."""
    if element_name != "button_text":
        return element_name
    if _step2_bucket_has_button_text_overlay(step2_bucket_names.get(bucket, "")):
        return element_name
    return _normalize_to_allowed(pre_el, ALLOWED_ELEMENTS) or "none"


def _bucket_element_after_gpt_and_bt_clamp(
    b: int,
    refined_per_bucket: Dict[int, str],
    element_per_bucket: Dict[int, str],
    step2_bucket_names: Dict[int, str],
) -> str:
    pre = element_per_bucket.get(b, "")
    if b in refined_per_bucket:
        el = _normalize_to_allowed(refined_per_bucket[b], ALLOWED_ELEMENTS)
    else:
        el = _normalize_to_allowed(pre, ALLOWED_ELEMENTS)
    return _clamp_button_text_without_detection(el, b, step2_bucket_names, pre)


def _normalize_to_allowed(raw: str, allowed: Set[str]) -> str:
    s = (raw or "").strip()
    if not s:
        return "none"
    if s in ("button_1", "button_2", "button_3", "button_4", "button_i", "button_bang"):
        s = "button"
    if s in allowed:
        return s
    low = s.lower()
    for a in allowed:
        if a.lower() == low:
            return a
    candidates = sorted(allowed)
    matches = difflib.get_close_matches(s.lower(), [c.lower() for c in candidates], n=1, cutoff=0.65)
    if matches:
        for c in candidates:
            if c.lower() == matches[0]:
                return c
    return "none"


def run_refinement(
    output_dir: str,
    model: str = "gpt-4o-mini",
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    video_path: Optional[str] = None,
) -> Dict[str, str]:
    output_dir = os.path.abspath(output_dir)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    resolved_video = _resolve_video_path(output_dir, video_path)
    frame_size: Optional[Tuple[int, int]] = None
    if resolved_video:
        frame_size = _video_frame_size(resolved_video)
        if frame_size:
            print(
                f"  Geometry clamp: video {resolved_video} -> {frame_size[0]}x{frame_size[1]}",
                flush=True,
            )
    ppt_rect = _load_ppt_rect_from_output_dir(output_dir)
    if frame_size and ppt_rect is None:
        print("  Geometry clamp: no debug/ppt_region.txt; clamp skipped.", flush=True)

    gaze_path = os.path.join(output_dir, "debug", "gaze_with_element.csv")
    if not os.path.isfile(gaze_path):
        raise FileNotFoundError(f"gaze_with_element.csv not found in {output_dir}/debug/")
    snapshot_path = os.path.join(output_dir, "debug", "ppt_ocr_snapshot.txt")
    ocr_snapshot = ""
    if os.path.isfile(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            ocr_snapshot = f.read()

    video_name = os.path.basename(os.path.normpath(output_dir))
    element_counts, total_rows, gaze_rows = _load_gaze_with_element(gaze_path)
    element_per_bucket = _build_element_per_bucket(gaze_rows, interval_sec)

    client = _get_openai_client()
    prompt1 = _build_prompt_ppt(video_name, ocr_snapshot, element_counts, total_rows)
    print(f"Calling OpenAI ({model}) for PPT element QA...")
    resp1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt1}],
        max_tokens=2048,
        temperature=0.2,
    )
    qa_content = (resp1.choices[0].message.content or "").strip()

    report_path = os.path.join(output_dir, "debug", "gaze_element_gpt_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(qa_content + "\n")

    refined_per_bucket: Dict[int, str] = {}
    if element_per_bucket:
        refine_prompt = _build_refine_elements_prompt(qa_content, element_per_bucket, interval_sec)
        if refine_prompt:
            print("Refining element names per time bucket (second GPT call)...")
            resp2 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": refine_prompt}],
                max_tokens=4096,
                temperature=0.1,
            )
            ref_content = (resp2.choices[0].message.content or "").strip()
            buckets_sorted = sorted(element_per_bucket.keys())
            refined_per_bucket = _parse_refined_elements(ref_content, buckets_sorted)

    step2_bucket_names = _load_step2_bucket_element_names(debug_dir)

    debug_csv = os.path.join(debug_dir, "step3_element_buckets.csv")
    with open(debug_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bucket",
                "time_start_sec",
                "time_end_sec",
                "element_before_gpt",
                "element_after_gpt_raw",
                "element_final",
            ]
        )
        for b in sorted(element_per_bucket.keys()):
            pre = element_per_bucket.get(b, "")
            post_raw = refined_per_bucket.get(b, "") if b in refined_per_bucket else ""
            final_el = _bucket_element_after_gpt_and_bt_clamp(
                b, refined_per_bucket, element_per_bucket, step2_bucket_names
            )
            w.writerow([b, b * interval_sec, (b + 1) * interval_sec, pre, post_raw, final_el])

    # Build final_gaze_table: per row use refined bucket label when available
    final_path = os.path.join(output_dir, "debug", "final_gaze_table.csv")
    with open(final_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["frame_idx", "timestamp_sec", "has_gaze", "gaze_x", "gaze_y", "element_name"],
        )
        writer.writeheader()
        for row in gaze_rows:
            frame_idx = row.get("frame_idx", "")
            ts = row.get("timestamp_sec", "")
            x_s = (row.get("x_bl") or "").strip()
            y_s = (row.get("y_bl") or "").strip()
            pre_el = (row.get("element_name") or "").strip() or "none"
            has_gaze = "1" if x_s and y_s else "0"
            try:
                t = float(ts)
            except (TypeError, ValueError):
                t = 0.0
            bucket = int(t / interval_sec) if interval_sec > 0 else 0
            if bucket in refined_per_bucket:
                element_name = _normalize_to_allowed(refined_per_bucket[bucket], ALLOWED_ELEMENTS)
            else:
                element_name = _normalize_to_allowed(pre_el, ALLOWED_ELEMENTS)
            if not element_name:
                element_name = "none"
            element_name = _clamp_button_text_without_detection(
                element_name, bucket, step2_bucket_names, pre_el
            )
            if (
                frame_size is not None
                and ppt_rect is not None
                and has_gaze == "1"
                and x_s
                and y_s
            ):
                try:
                    x_tl, y_tl = bl_to_tl(float(x_s), float(y_s), frame_size[1])
                    if x_tl is not None and y_tl is not None:
                        element_name = _clamp_gpt_label_to_outside_ppt_policy(
                            float(x_tl),
                            float(y_tl),
                            ppt_rect,
                            frame_size[0],
                            frame_size[1],
                            element_name,
                        )
                except (TypeError, ValueError):
                    pass
            writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "timestamp_sec": ts,
                    "has_gaze": has_gaze,
                    "gaze_x": x_s,
                    "gaze_y": y_s,
                    "element_name": element_name,
                }
            )

    final_txt = os.path.join(output_dir, "final_gaze_table.txt")
    _write_final_gaze_txt(final_path, final_txt)

    return {
        "final_gaze_table": final_path,
        "final_gaze_table_txt": final_txt,
        "gpt_report": report_path,
        "debug_element_buckets": debug_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PPT: GPT refinement for gaze element_name labels.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Source .mp4 (for frame size + outside-PPT clamp). Default: data/video_image/<output_dir_basename>.mp4",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    args = parser.parse_args()
    stats = run_refinement(
        args.output_dir,
        model=args.model,
        interval_sec=args.interval,
        video_path=args.video,
    )
    print("Wrote", stats["final_gaze_table"], "and", stats.get("final_gaze_table_txt", ""))
    print("Report:", stats["gpt_report"])


if __name__ == "__main__":
    main()
