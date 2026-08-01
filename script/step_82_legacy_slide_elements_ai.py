"""Legacy AI slide-element helper.
Detect **PPT-slide** circular ``button`` boxes and ``button_text`` overlays via OpenAI vision (VLM).

**Buttons:** bordered circles with one digit/symbol on the slide (maybe **none**). After a click,
the same control often shows **green solid fill** — that state and the **button_text**
overlay are a **pair** (overlay appears **on top of** the slide, usually **after** a button click).

**button_text:** a **layer over** the PPT area (modal/frame), not static slide copy.

Crop is slide-only. Model: ``OPENAI_PPT_UI_MODEL`` (default ``gpt-4o``). Requires ``OPENAI_API_KEY`` or ``.env``.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore
    load_dotenv = None

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MAX_SIDE = 1800
_MODEL = os.environ.get("OPENAI_PPT_UI_MODEL", "gpt-4o").strip() or "gpt-4o"

# Import Element from step_2 only when building results — avoid circular import at module load
_Element = None


def _client() -> Optional[Any]:
    if OpenAI is None:
        return None
    env_path = os.path.join(_ROOT, ".env")
    if load_dotenv is not None and os.path.isfile(env_path):
        load_dotenv(env_path)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def _parse_ui_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if "```" in t:
        for ch in t.split("```"):
            ch = ch.strip()
            if ch.lower().startswith("json"):
                ch = ch[4:].strip()
            if ch.startswith("{"):
                t = ch
                break
    try:
        i = t.index("{")
        j = t.rindex("}") + 1
        return json.loads(t[i:j])
    except (ValueError, json.JSONDecodeError):
        return None


def _label_to_internal(label: str) -> Optional[str]:
    s = (label or "").strip().lower()
    if s in ("1", "2", "3"):
        return f"button_{s}"
    if s in ("i",):
        return "button_i"
    if s in ("!",):
        return "button_bang"
    if s in ("4",):
        return "button_4"
    return None


def _elements_from_parsed(
    obj: Dict[str, Any],
    px0: int,
    py0: int,
    sx: float,
    sy: float,
    fw: int,
    fh: int,
    slide_py0: float,
    slide_py1: float,
    slide_px0: float,
    slide_px1: float,
) -> List[Any]:
    global _Element
    if _Element is None:
        from step_94_legacy_slide_detect_elements import Element as _E

        _Element = _E

    out: List[Any] = []
    buttons = obj.get("buttons") or []
    if isinstance(buttons, list):
        for b in buttons:
            if not isinstance(b, dict):
                continue
            lab = _label_to_internal(str(b.get("label", "")))
            if lab is None:
                continue
            try:
                x0 = float(b["x0"]) * sx + px0
                y0 = float(b["y0"]) * sy + py0
                x1 = float(b["x1"]) * sx + px0
                y1 = float(b["y1"]) * sy + py0
            except (KeyError, TypeError, ValueError):
                continue
            x0 = max(0.0, min(float(fw - 1), x0))
            x1 = max(0.0, min(float(fw - 1), x1))
            y0 = max(0.0, min(float(fh - 1), y0))
            y1 = max(0.0, min(float(fh - 1), y1))
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                continue
            # PPT slide only; drop anything outside slide vertical bounds (e.g. nav hallucinations).
            if y0 < slide_py0 - 2 or y1 > slide_py1 + 2:
                continue
            bw, bh = x1 - x0, y1 - y0
            if bh <= 1 or bw / bh < 0.65 or bw / bh > 1.45:
                continue
            slide_w = max(1.0, float(slide_px1 - slide_px0))
            slide_h = max(1.0, float(slide_py1 - slide_py0))
            slide_area = slide_w * slide_h
            if bw * bh > 0.0035 * slide_area or bw * bh < 400.0:
                continue
            out.append(_Element(lab, x0, y0, x1, y1))

    frames = obj.get("button_text_frames") or obj.get("button_text") or []
    if isinstance(frames, list):
        for f in frames:
            if not isinstance(f, dict):
                continue
            try:
                x0 = float(f["x0"]) * sx + px0
                y0 = float(f["y0"]) * sy + py0
                x1 = float(f["x1"]) * sx + px0
                y1 = float(f["y1"]) * sy + py0
            except (KeyError, TypeError, ValueError):
                continue
            x0 = max(0.0, min(float(fw - 1), x0))
            x1 = max(0.0, min(float(fw - 1), x1))
            y0 = max(0.0, min(float(fh - 1), y0))
            y1 = max(0.0, min(float(fh - 1), y1))
            if x1 <= x0 + 4 or y1 <= y0 + 4:
                continue
            # Drop static titles / instruction blocks mislabeled as popups (wide band in top ~12% of slide).
            sw = max(1.0, float(slide_px1 - slide_px0))
            sh = max(1.0, float(slide_py1 - slide_py0))
            fw_ = x1 - x0
            if fw_ > 0.82 * sw and y0 < slide_py0 + 0.12 * sh:
                continue
            if fw_ * (y1 - y0) > 0.40 * sw * sh:
                continue
            if y1 > slide_py1 + 6.0:
                continue
            out.append(_Element("button_text", x0, y0, x1, y1))

    return out


def detect_ppt_ui_elements_openai(
    frame_bgr: np.ndarray,
    ppt_rect: Tuple[float, float, float, float],
) -> Optional[List[Any]]:
    """
    Returns a list of `Element` (button_* and button_text) in **full-frame** coordinates,
    or ``None`` if the API is unavailable or the response could not be parsed.

    A successful response with no buttons/frames yields ``[]`` (caller may fall back to CV).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    fh, fw = frame_bgr.shape[:2]
    px0, py0, px1, py1 = [int(round(v)) for v in ppt_rect]
    px0 = max(0, min(fw - 1, px0))
    py0 = max(0, min(fh - 1, py0))
    px1 = max(0, min(fw - 1, px1))
    py1 = max(0, min(fh - 1, py1))
    if px1 <= px0 + 8 or py1 <= py0 + 8:
        return None

    # PPT slide only (y1 = top of red nav); buttons live on the slide, not in the strip.
    crop_y0 = py0
    crop_y1 = py1
    crop = frame_bgr[crop_y0 : crop_y1 + 1, px0 : px1 + 1]
    ch, cw = crop.shape[:2]
    scale = min(_MAX_SIDE / max(cw, ch), 1.0)
    if scale < 1.0:
        nw, nh = int(round(cw * scale)), int(round(ch * scale))
        small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        small = crop
        nw, nh = cw, ch

    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok or buf is None:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    client = _client()
    if client is None:
        return None

    prompt = """This image is **only the PPT/slide content area** (white/colored slide). The bottom red navigation strip is **not** included.

**Product behavior (use this to disambiguate):**
- **button_text** is an **overlay layer drawn on top of** the slide — a framed text/modal area. It is **not** part of the base slide master. It typically **only appears after** the learner **clicks** one of the circular lesson controls.
- **Buttons** are **bordered circles** with **exactly one** glyph (**1**, **2**, **3**, **4**, **i**, **!**). After a click, the **same** control is shown with **green solid fill** inside the circle (not a thin outline or outer glow). The **green-filled** button and the **button_text overlay** in the same frame are a **pair** (same interaction). Always box the **full** circle, including when it is solid green.

1) **buttons** — There may be **none**. Count only **bordered** circular (or near-circular) controls on the slide with **one** matching glyph. A clicked control has **green solid fill** — treat it as the same button type. **Never** place a button box on **words inside a sentence** (e.g. the word "test" in "test your knowledge", or "the") — those are **not** buttons. **Never** label headings, paragraphs, bullets, diagram pie wedges, or unbordered shapes.

2) **button_text_frames** — **Only** if you see a **separate overlay** on top of slide content: e.g. **dark border** + **light interior**, reading/modal layer. If the frame shows **only** static slide text (titles, instructions) with **no** overlay layer, use `"button_text_frames": []`. Do not treat base slide copy as button_text.

Coordinates: pixel floats in **this image** (top-left origin, x right, y down). Tight boxes.

Return **only** valid JSON (no markdown):
{
  "buttons": [ {"label": "1"|"2"|"3"|"4"|"i"|"!", "x0": float, "y0": float, "x1": float, "y1": float} ],
  "button_text_frames": [ {"x0": float, "y0": float, "x1": float, "y1": float} ]
}

Use empty arrays when absent. Omit uncertain detections."""

    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                        },
                    ],
                }
            ],
            max_tokens=3500,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

    obj = _parse_ui_json(raw)
    if obj is None:
        return None

    sx = cw / float(nw)
    sy = ch / float(nh)
    slide_py0_f = float(py0)
    slide_py1_f = float(py1)
    slide_px0_f = float(px0)
    slide_px1_f = float(px1)
    return _elements_from_parsed(
        obj,
        px0,
        crop_y0,
        sx,
        sy,
        fw,
        fh,
        slide_py0_f,
        slide_py1_f,
        slide_px0_f,
        slide_px1_f,
    )
