# PPT gaze → element policy (authoritative)

This document matches **`scripts_image/step_2_detect_elements.py`** (`choose_element_for_gaze`, detectors, and helpers). Step 3 (GPT) and step 4 (validation) add behavior described in the last sections.

**Exported `element_name` values:** `heading`, `paragraph`, `button_text`, `button`, `image`, `navigation_bar`, `blank_area`, `none` (no gaze). Internal strip labels `button_1` … `button_bang` map to public **`button`**.

---

## Pipeline context

| Step | Role |
|------|------|
| **1** | Gaze coordinates per frame (bottom-left → top-left for labeling). |
| **2** | PPT rectangle + UI elements per time bucket; assign `element_name` per gaze row. |
| **3** (optional) | GPT may **overwrite one label per time bucket** for all rows in that bucket; then a **geometry clamp** may fix slide labels when gaze is outside the **inset** PPT box (see below). |
| **4** (optional) | Validation images: blue **PPT** outline uses an **inset** of the raw PPT rect — the drawn box can be slightly tighter than the rect used in step 2. |

---

## A. Region definitions

### A1 — PPT vs no PPT (step 2 geometry test)

- **PPT:** Gaze point inside the axis-aligned **PPT rectangle** `(x0, y0, x1, y1)` in **top-left** coordinates. This rect is from chrome / OpenAI / CV (see code); bottom `y1` is intended near the **top of the red navigation bar**.
- **No PPT:** Anywhere else (pillarbox, letterbox, browser chrome, etc.).

### A2 — No PPT: allowed labels only

Evaluate **in order**:

1. **`navigation_bar`** — gaze on/near the bottom **red navigation strip** (heuristics: y near slide bottom, distance to nav bar element, bottom band).
2. **`blank_area`** — everything else outside PPT.

No slide content labels (`heading`, `paragraph`, `button_text`, `button`, `image`) apply outside PPT.

### A3 — Inside PPT: navigation strip first

If the gaze matches **nav strip** heuristics (loose PPT bottom vs real bar), assign **`navigation_bar`** before any slide UI — even if the stored PPT rect would still include the strip.

---

## B. How UI elements are detected (per bucket frame)

These populate the `elements` list; gaze logic consumes them.

| Element | Detection (summary) |
|--------|----------------------|
| **`button_text` (popup)** | **Overlay layer on top of** the slide (not base slide text). In the product it usually appears **after** a circular lesson **button** is clicked; **green solid fill** in the clicked button and **button_text** overlay are a **pair**. Detectors: thin dark border (Canny) → bright white/cream blob → green HSV blob; optional OpenAI VLM. If **previous bucket frame** exists: mean grayscale absdiff on PPT crop ≥ threshold ⇒ **inferred content change**; then estimate popup from **diff mask**. Log: `debug/ppt_content_change.csv`. |
| **heading** | Red title in top band of PPT crop (HSV + BGR). |
| **paragraph** | Adaptive threshold text lines; excludes dilated red + **masked-out popup** interior. |
| **button** | Bordered **circles** with one symbol (`i`, `!`, `1`–`4`); clicked = **green solid fill** inside the circle (EasyOCR strip and/or OpenAI VLM on slide crop). |
| **image** | Canny regions in PPT, chromatic patch, overlap gating vs **heading/paragraph/buttons** — **not** vs `button_text` popup (so diagrams under popups can exist). |
| **navigation_bar** | Per-frame nav top → full-width strip to frame bottom. |

---

## C. Gaze assignment inside PPT (strict priority)

After **A3** (nav strip), apply **in this order**. Earlier steps win; do not skip to a later step unless earlier steps do not apply.

### C1 — `button_text` (overlay / text frame)

- **Semantics:** `button_text` is an **overlay** on the PPT; in this UI it is commonly shown **together with** a **green solid-filled** clicked **button** — treat as paired content when both appear in the bucket frame.
- **Detector required:** a **`button_text`** label is only possible if step 2 actually added a **`button_text`** element (detector box) for that bucket. There is **no** distance/padding fallback: gaze must lie **inside** that axis-aligned box.
- **Nav exception:** if gaze is in the nav zone for these checks, **`navigation_bar`** instead.
- **Pixel gate:** assign **`button_text`** only if the local gaze patch looks **neutral** (white / black / unsaturated gray — “paper + ink”). If the patch is **saturated** (e.g. diagram colors), **do not** assign `button_text`; continue. If `frame_bgr` is missing, **do not** assign `button_text`.

Step 3 (GPT) cannot keep **`button_text`** unless **`step2_element_buckets.csv`** lists **`button_text`** among detector names for that bucket.

### C2 — `button` (strip circles: `i`, `!`, `1`–`3`)

- Nearest strip button within threshold (tie-break: `TIE_PRIORITY`).
- Else gaze **inside** a strip button box.

### C3 — Chromatic background → `heading` or `image` only

If **frame** is available: small patch around gaze. If patch is **not** neutral (not white/black/gray):

- Inside **heading** box → **`heading`**
- Inside **image** box → **`image`**
- Else if in **title band** and patch reads **red heading** → **`heading`**
- Else → **`image`**

### C4 — `image` (detector box)

Gaze strictly inside an **`image`** element rectangle.

### C5 — Distance match (remaining elements)

Nearest element within threshold among **`heading`**, **`paragraph`** (not `button_text`, not strip buttons, not `image` in this loop). Tie-break: **`TIE_PRIORITY`**:

`heading` → `paragraph` → `button_1`…`button_bang` → `button_text` → `image` → `navigation_bar`

*(Strip buttons are already handled in C2.)*

### C6 — Point inside any remaining box

Any element not excluded (e.g. paragraph, heading) — **not** `button_text` / strip buttons here.

### C7 — Nearest fallback

If still on slide (y above slide bottom), nearest element within **`GAZE_NEAREST_FALLBACK_PX`**, excluding `navigation_bar`, `image`, `button_text`, strip buttons.

### C8 — `blank_area`

Default inside PPT when nothing else matches.

---

## D. Post-pass: `_finalize_gaze_label`

If the candidate is a **slide content** label but gaze is **strictly below** the effective slide bottom (`slide_bottom` = min of PPT `y1` and detected nav top), remap to **`navigation_bar`** or **`blank_area`** so slide UI is never assigned on the chrome/nav strip when geometry is loose.

---

## E. Step 3 (optional GPT)

- Builds **one label per time bucket** from GPT; applies that label to **every gaze row** in the bucket — **ignores per-frame (x, y)**.
- **Clamp:** If gaze (with video + `ppt_region.txt`) is **outside** the **inset** PPT box (same idea as step 4 blue outline) and the label is a slide type, force **`navigation_bar`** or **`blank_area`** — so GPT cannot turn nav gaze into `button_text` for the whole bucket.

To preserve step 2 spatial logic, skip step 3 or treat GPT output as advisory only.

---

## F. Step 4 validation

- **Blue “PPT area”** is drawn with an **inset** on the raw PPT rect; gaze can look “outside” blue while still inside raw rect for step 2.
- Gaze dot label comes from **`final_gaze_table`** (after step 3 if run).

---

## G. Constants (tune in code)

- `DEFAULT_GAZE_MATCH_THRESHOLD_PX` — distance to element edge.
- `GAZE_NEAREST_FALLBACK_PX` — nearest-region fallback.
- `PPT_CONTENT_CHANGE_MEAN_ABS_DIFF_THR` — bucket-to-bucket PPT change → diff-based popup.
- `GAZE_BG_PATCH_RADIUS` — patch for neutral vs chromatic and `button_text` pixel gate.

---

## Summary order (inside PPT, after nav)

1. **`button_text`** (geometry + neutral patch if frame available)  
2. **`button`** (circles)  
3. **Chromatic bg** → **`heading`** or **`image`**  
4. **`image`** (box hit)  
5. **Distance** → heading / paragraph / …  
6. **Point-in-box** → remaining  
7. **Nearest fallback**  
8. **`blank_area`**
