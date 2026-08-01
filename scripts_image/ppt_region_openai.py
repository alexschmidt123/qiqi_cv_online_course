"""
One-shot PPT content rectangle via OpenAI vision (gpt-4o-mini).

The rectangle uses top-left pixel coordinates (y down):
- Left: right edge of left black pillarbox
- Right: left edge of right pillarbox (same bar height as left)
- Top: same horizontal line as the top of both side bars
- Bottom: top edge of the red navigation bar

Used once on a reference frame; the same rect is reused for all buckets.
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore
    load_dotenv = None

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_MAX_SIDE = 1280
_MODEL = "gpt-4o-mini"


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


def _parse_json_rect(text: str) -> Optional[Dict[str, float]]:
    t = (text or "").strip()
    if "```" in t:
        chunks = t.split("```")
        for ch in chunks:
            ch = ch.strip()
            if ch.lower().startswith("json"):
                ch = ch[4:].strip()
            if ch.startswith("{"):
                t = ch
                break
    try:
        i = t.index("{")
        j = t.rindex("}") + 1
        obj = json.loads(t[i:j])
        return {
            "x0": float(obj["x0"]),
            "y0": float(obj["y0"]),
            "x1": float(obj["x1"]),
            "y1": float(obj["y1"]),
        }
    except (ValueError, KeyError, json.JSONDecodeError, TypeError):
        return None


def detect_ppt_rect_from_frame_openai(frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (x0,y0,x1,y1) in original frame pixels, or None if API unavailable / parse failed.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    scale = min(_MAX_SIDE / max(w, h), 1.0)
    if scale < 1.0:
        nw, nh = int(round(w * scale)), int(round(h * scale))
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr
        nw, nh = w, h

    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok or buf is None:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    client = _client()
    if client is None:
        return None

    prompt = """This image is one frame from a screen recording of a presentation slide.

Find the axis-aligned RECTANGLE of the slide/PPT content area (the main white or slide region where the lesson content is shown). Use these geometric rules:
- LEFT edge: the vertical line at the RIGHT side of the left black letterbox/pillarbox bar.
- RIGHT edge: the vertical line at the LEFT side of the right black letterbox bar (the two side bars have the same height and similar width).
- TOP edge: the horizontal line that aligns with the TOP of both side black bars (the top border of the slide content).
- BOTTOM edge: the TOP edge of the RED navigation/control bar at the very bottom of the screen (do not include that red bar inside the rectangle).

The box must be TIGHT: only the white/slide pixels — exclude all black letterbox. If unsure, shrink a few pixels inward from the inner edges so the blue outline would sit just inside the bars.

Return ONLY valid JSON on one line with floating-point pixel coordinates in THIS image coordinate system (origin top-left, x right, y down), keys exactly:
{"x0":..., "y0":..., "x1":..., "y1":...}
where (x0,y0) is top-left and (x1,y1) is bottom-right of the slide area, inclusive."""

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
            max_tokens=500,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

    rect = _parse_json_rect(raw)
    if rect is None:
        return None

    sx = w / float(nw)
    sy = h / float(nh)
    x0 = rect["x0"] * sx
    y0 = rect["y0"] * sy
    x1 = rect["x1"] * sx
    y1 = rect["y1"] * sy

    x0 = max(0.0, min(x0, float(w - 1)))
    x1 = max(0.0, min(x1, float(w - 1)))
    y0 = max(0.0, min(y0, float(h - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return None
    return x0, y0, x1, y1
