# SOP: ChatGPT API for slide element segmentation

**Single source of truth** for offline reference-pattern creation (ChatGPT) and how online gaze uses those frozen patterns.

Use this file when:
- building the library the first time,
- **adding new slides / popup variants later**,
- calling the ChatGPT vision API by hand or from code.

Machine prompt for the API = section **API prompt** (between HTML markers). Code loads that block via `script/step_2_analyze_video_with_ai.py` → `_load_slide_segmentation_sop()`.

Offline multi-video builder: `script/offline_build_slide_library.py`

---

## 0. Offline vs online (core split)

| Job | When | Who | Goal |
|-----|------|-----|------|
| **Get reference patterns** | **Offline** | ChatGPT vision + human QA | For each content state, produce one reference image + one element box table |
| **Use reference patterns** | **Online** (every video / every gaze) | OpenCV match + geometry only | Find the right reference pattern → locate gaze → record element for that gaze |

**Online does only this:**

```text
find right reference pattern (page + variant)
        → locate gaze point on that pattern
        → record element information for this gaze
```

Online does **not** invent boxes, call ChatGPT, or build new patterns.  
If no reference matches → queue for **offline** library update; do not segment live.

**Offline does only this:** decide content states, pick canonical references, run ChatGPT once per reference, QA, freeze CSVs/PNGs.

**Offline must ignore (never treat as content, never box, never use to define a variant):**

- Red gaze circle (gaze point)
- Mouse / hand **cursor**

Strip both from crops **before** clustering, before saving the canonical reference, and before the ChatGPT call. Two frames that differ only by gaze or cursor are the **same** variant.

ChatGPT’s entire role in this project is the **offline** row. It never “learns” gaze online.

**All participant videos for this course share the same slides** → one shared library under `data/slide_standard_library/`. New users do not need new patterns unless the course content changes.

---

## 1. Online workflow (no ChatGPT)

```text
video frame + red gaze
        │
        ├─① detect gaze (x, y)                         [OpenCV]
        ├─② locate slide area (course_roi) → crop      [OpenCV]
        ├─③ strip gaze/cursor for matching             [OpenCV]
        ├─④ find slide PAGE + VARIANT → reference      [match frozen refs]
        ├─⑤ load that reference’s element pattern      [frozen CSV]
        ├─⑥ map gaze into reference coordinates
        ├─⑦ hit-test → learning element (priority)
        └─⑧ record / aggregate element history
              {time, slide_page, slide_variant, element, duration}
```

**Rules**

- Matching ignores gaze spot and cursor; content state only.
- **One content state ↔ one reference image ↔ one element table.**
- Gaze/cursor-only crops are not new variants.

---

## 2. Library identity model

| ID | Meaning | Example |
|----|---------|---------|
| `slide_page` | Logical lesson page (pagination / base layout) | “Explore the Cell Cycle” base diagram |
| `slide_variant` | Visual state on that page | base, button-1 popup, button-2 popup, … |
| `reference` | Canonical **cleaned** crop for that state | `slide_015_ref_01.png` |
| `elements` | Boxes for **that** reference only | `slide_015.csv` |

Current on-disk shape (compatible with runtime matcher):

```text
data/slide_standard_library/
  slides.csv                 # slide_id, description, reference_images, template_image, element_table
  references/<slide_id>_ref_01.png   # prefer ONE ref per state
  elements/<slide_id>.csv            # boxes for that ref only
  templates/<slide_id>.png           # QA overlay only (not used online)
```

**Hard rule:** never attach many different popup crops as `ref_02…ref_N` under **one** element CSV. Each popout state needs its own `slide_id` (or explicit page+variant ids) and its own CSV.

Legacy bug to avoid: many `ref_*.png` (gaze/cursor/popout mix) + a single incomplete CSV.

---

## 3. When to use ChatGPT (offline only)

| Stage | ChatGPT? |
|-------|----------|
| Online: match reference, gaze, element history | **No** |
| Offline: label elements on one cleaned reference crop | **Yes** (API prompt below) |
| Offline: after a new unmatched state is approved | **Yes** once → freeze → online can match it |

---

## 4. Offline prepare (build / extend library)

### 4.1 Steps

```text
1. Sample frames from course video(s) (sequential read; ~1–2 s interval)
2. Locate course_roi (pink control strip → white slide viewport)
3. REMOVE red gaze circle + cursor (inpaint) — mandatory
4. Cluster into content states (ignore gaze/cursor-only diffs)
5. Pick one clean canonical reference per state
6. ChatGPT labels THAT cleaned reference only (API prompt below)
7. Post-process + human QA on template overlay
8. Freeze reference PNG + elements CSV for that state
```

Popup / green-button states = **new variants**, each with its own reference and element list (include the popup box + clicked button on that variant). Gaze/cursor position do **not** create variants.

### 4.2 Tech details that matter later (new slides)

#### A. Slide area (`course_roi`)

- Implemented in `_locate_content_roi` (`script/step_2_analyze_video_with_ai.py`).
- Find widest **pink** HSV band in the lower half of the full UI frame; top of that band ≈ bottom of the white slide; derive top from a stable aspect above the controls.
- Fallback fractions if pink detection fails.
- ChatGPT must receive **only this crop**, not browser chrome / page title above the player.

#### B. Strip overlays before any AI / clustering

Function: `_remove_overlay_markers` = red-gaze inpaint + white cursor inpaint.

| Marker | How | Do not remove |
|--------|-----|----------------|
| Red gaze circle | HSV red mask; prefer neighborhood of known gaze; else small red blobs only | Maroon/pink **title text** |
| White / light cursor | Bright low-saturation blobs of cursor size | White **hand glyph inside green clicked buttons** (exclude green neighborhood) |

If gaze/cursor remain in the image sent to ChatGPT, boxes drift and false “elements” appear.

#### C. Clustering / uniqueness (OpenCV)

- Compare **cleaned ROI crops** with mean absdiff on resized grayscale (`_crop_difference`, ~320×180).
- **Per-video discovery** threshold (keep popups): about `change_threshold ≈ 0.7`.
- **Cross-video merge** (same course, all participants): start around `merge_threshold ≈ 1.2` for strict uniqueness; expect over-segmentation. Empirically on this course’s 7 videos:

| `merge_threshold` | Approx. unique states |
|-------------------|------------------------|
| 1.2 | ~110 |
| 2.0 | ~85 |
| 3.0 | ~68 |
| 4.0 | ~55 |

- Prefer **fewer false merges** (missed popups) over aggressive merge; then manually drop gaze-noise duplicates while QA’ing previews.
- Preview-only command (no API spend):

```bash
conda activate cv_env
PYTHONUNBUFFERED=1 python3 script/offline_build_slide_library.py \
  --cluster-only --sample-interval 2.0 --merge-threshold 1.2
# writes data/slide_library_cluster_preview/slide_XXX.png
```

#### D. One ChatGPT call = one state

- Model: **`gpt-4o`** (avoid mini for freezing production boxes).
- Temperature: **`0`**.
- Input text: **API prompt** section below (exact).
- Input image: one cleaned reference JPEG/PNG, `detail: "high"`.
- Output: JSON only (`state_description` + `elements[]` with `x0,y0,x1,y1`).
- Retries: up to **3** if no non-blank elements.
- Cost note: each unique state ≈ one vision call; cluster first, QA preview, then label.

#### E. Post-process (code, mandatory after every API response)

1. Clamp coords to `[0,1]`; ensure `x0 < x1`, `y0 < y1`.
2. Force **AABB** for all non-button types (collapse skewed quads).
3. Optional: near-square `button` → ellipse.
4. Append `blank_area` `[0,0]–[1,1]` priority −100 if missing.
5. Reject/retry if title box sits in empty white, or `image` is tiny / whole-crop incorrectly.

#### F. Full offline build (all videos → shared library)

```bash
conda activate cv_env
PYTHONUNBUFFERED=1 python3 script/offline_build_slide_library.py \
  --video-dir data/video_slide \
  --standard-library-dir data/slide_standard_library \
  --model gpt-4o \
  --sample-interval 2.0 \
  --change-threshold 0.7 \
  --merge-threshold 3.0 \
  --workers 3
```

Backs up previous library to `data/slide_standard_library_backup/` unless `--no-backup`.

#### G. Adding **new slides** later (incremental)

1. Capture video (or stills) that include the new page / popup.
2. Run `--cluster-only` against new footage **or** match new crops to existing refs.
3. States with no library match → new `slide_id`s.
4. For each new cleaned reference only:
   - Call ChatGPT with **API prompt** + that image.
   - QA template.
   - Append row to `slides.csv`; write `references/` + `elements/` + `templates/`.
5. Do **not** re-label unchanged states.
6. If the player chrome / pink bar / aspect changes, re-check `_locate_content_roi` before labeling.

#### H. Manual one-off API call (ChatGPT UI or raw API)

1. Export one cleaned crop (no gaze, no cursor, content ROI only).
2. Paste the **API prompt** text as the instruction.
3. Attach the image.
4. Copy JSON → convert `x0,y0,x1,y1` into the element CSV schema (`shape=rectangle` + `x_min…y_max`, or ellipse fields for buttons).
5. Draw overlay; run Human QA (§7).

CSV columns expected by the loader:

```text
element_id,decision,element_type,priority,shape,
x_min,x_max,y_min,y_max,center_x,center_y,radius_x,radius_y,points
```

---

## 5. Input / taxonomy / geometry (summary)

**Input:** one content crop; coords `[0,1]` top-left relative to that crop; one image = one content state.

**Types only:** `slide_title` | `paragraph` | `image` | `button` | `popup` | `slide_navigation` | `blank_area`

**Priorities:** blank −100 · title 10 · button 20 · popup ≥100 · body over image → raise text

**Geometry:** AABB `x0,y0,x1,y1` for non-buttons; text pad ≈1em; full images; circular glyph buttons only; popup only if framed overlay visible; always `blank_area`.

**Labels:** quote visible content, not type names.

---

## 6. API call (canonical)

| Field | Value |
|-------|--------|
| Model | `gpt-4o` |
| Temperature | `0` |
| Text | **API prompt** below |
| Image | one cleaned reference, `detail: high` |
| Output | JSON only |

See §4.2 D–E for retries and post-process.

---

## 7. Human QA

Fail if:

- [ ] Title / text boxes in empty margin or on the wrong ink  
- [ ] Text pad ≫ 1em  
- [ ] Photo/diagram cut off  
- [ ] Skewed / diamond boxes  
- [ ] Missing circular `1–4` / `i` / `!` buttons when visible  
- [ ] Framed overlay present but not labeled `popup`  
- [ ] Popup variant missing clicked green `button`  
- [ ] Non-SOP types (`web_panel`, `panel`, …)  
- [ ] Red gaze or cursor boxed  
- [ ] Missing `blank_area`  
- [ ] One CSV reused for different popup states  

Pass → freeze that state’s reference + element CSV + template.

---

## 8. Element history (online output)

```text
time_point, slide_page, slide_variant, element, duration
```

(or today’s flattened `slide_id` + `element` until page/variant ids are fully wired)

Gaze outside course ROI → `unrelated_content`.

---

## 9. Known failure modes (from library QA)

Recorded so later ChatGPT runs avoid repeating them:

1. **Incomplete elements** — interactive diagrams labeled title+instruction only; numbered hotspots omitted.  
2. **Shared CSV across popouts** — refs showed G1/S/Mitotic popups but one element table.  
3. **Gaze/cursor treated as state** — many near-duplicate refs.  
4. **Boxes floating above content** — AI coords not aligned to cleaned crop; always QA overlay on the **same** PNG you freeze as reference.  
5. **Non-SOP types** — `web_panel` / `panel`; map to `popup` or `paragraph`/`image`.  
6. **Full-crop `image`** — useless vs `blank_area`; reject.  
7. **Title priority 0 / huge title band** — prefer priority 10 and ~1em pad.

---

## API prompt

Code extracts everything between the markers. Edit **only** this block when changing what ChatGPT sees. For new slides later, keep this prompt unless taxonomy/geometry changes (then update SOP + prompt together).

<!-- BEGIN_API_PROMPT -->
You are labeling learning elements on ONE cropped online-course slide image for a frozen attention library.

This image is exactly one (slide_page, slide_variant) reference. Label only what is visible in THIS image. Use the same taxonomy and geometry for every future page/variant.

INPUT
- Image = slide content crop only (top-left origin, x right, y down).
- Coordinates are normalized floats in [0,1] relative to THIS image.
- One image = one visual state (base page OR one popup/click variant).
- Do not invent browser chrome, page headers, or UI that is not visible in the crop.
- Ignore and do not box: red gaze circle, mouse/hand cursor (treat as absent).
- Do not treat gaze or cursor position as a reason to add elements.

ELEMENT TYPES (only these; omit a type if absent)
- slide_title: main slide heading (often large colored title text)
- paragraph: instruction lines, body copy, text columns, captions (one box per coherent block)
- image: full photo, diagram, figure, or illustration extent
- button: bordered circle with exactly one glyph such as {1,2,3,4,i,!} (green solid fill = clicked)
- popup: framed overlay panel (dark border + light interior) that appears after interaction
- slide_navigation: pagination dots (or in-crop player controls if clearly visible)
- blank_area: entire image [0,0]-[1,1], priority -100

GEOMETRY
- For slide_title, paragraph, image, popup, slide_navigation, blank_area: axis-aligned rectangles only.
  Return x0,y0,x1,y1 with x0<x1 and y0<y1. Forbidden: skewed/tilted/parallelogram quads.
- Text (slide_title, paragraph): pad ~1em around glyph ink — tight, not a large empty band.
  Do not place a title box in empty white away from the real title.
  Instruction paragraph sits just under the title; minimize overlap.
- image: include the FULL visible figure; do not cut off edges; stop at a neighboring text panel edge.
- button: box the full circle; never label words inside sentences as buttons; never label pie wedges or pagination dots as buttons.
- popup: only if a separate overlay layer is visible; priority >= 100; keep clicked green button + popup in the same state when both are visible.
- Always include blank_area covering the whole image.

PRIORITY
- blank_area: -100
- slide_title: 10
- button: 20
- popup: >= 100
- overlapping body text over image: give text/popup higher priority than image

LABELS
- Human-readable content of the region (quote the text or short description), not the type name.

COVERAGE
- Label every meaningful learning surface on this crop for THIS variant only.
- New layouts still use ONLY the types above.
- If unsure about an element, omit it rather than guessing a bad box.

OUTPUT
Return ONLY valid JSON (no markdown):
{
  "state_description": "short description of this slide page variant",
  "elements": [
    {
      "element_type": "slide_title|paragraph|image|button|popup|slide_navigation|blank_area",
      "label": "string",
      "priority": 0,
      "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0
    }
  ]
}
<!-- END_API_PROMPT -->
