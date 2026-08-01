# Online-course attention analysis

This project converts an online-course screen recording into a table describing
what the participant looked at and for how long.

## Pipeline

All Python code is in `script/` and indexed by execution order:

1. `step_1_detect_gaze.py` — pure OpenCV/NumPy red gaze-point detection. It does
   not use OCR or AI.
2. `step_2_analyze_video_with_ai.py` — OpenCV representative-state detection,
   EasyOCR text evidence, AI video classification/layout interpretation, element
   library generation, per-video report, and annotated layout images.
3. `step_3_map_gaze_to_events.py` — deterministic point-in-polygon matching and
   duration aggregation.

Files numbered `80–99` are consolidated legacy helpers and tests retained for
reproducibility. They are not called by the default production runner.

## Supported videos

### Scrollable article in a website

The AI reconstructs the full article across scrolling screenshots and reports
the article title, full text, reading order, and layout labels. Website elements
such as the navigation bar, page title, panels, paragraphs, and images are also
stored as attention regions.

### Slides in a website

The AI identifies all distinct slide states. If clicking a button reveals new
content, the revealed version is a separate state even when the base slide is
unchanged. Each state gets an annotated PNG under `layout/`:

- the cleaned screenshot is rendered at 50% opacity over white;
- different colors outline titles, paragraphs, images, buttons, popups, website
  navigation, and slide navigation;
- labels identify the specific content represented by each region.

AI determines semantic patterns and layout. EasyOCR plus OpenCV determine the
actual text hit regions used for gaze matching.

## Installation

```bash
conda create -n cv_env python=3.11 -y
conda activate cv_env
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

The default AI model is `gpt-5.6-sol`. Override it with `-model` when needed.

## Run

Automatically classify each video:

```bash
bash run.sh -dir data/video
```

Process one known article or slide video:

```bash
bash run.sh -dir data/video/R4,P1_1.mp4 -type article
bash run.sh -dir data/video_image/R4_P1.mp4 -type slides
```

Useful options:

```text
-type auto|article|slides
-model MODEL
-sample-interval SECONDS
-min-duration SECONDS
```

## Output

Each video writes to `output/<video-name>/`:

- `attention_table.csv` — final per-user attention events;
- `ai_report.md` and `ai_report.json` — AI analysis for that video;
- `element_library.json` — timed semantic polygons used for gaze matching;
- `layout/*.png` — 50%-opacity annotated layout states;
- `gaze_coordinates.csv` — raw OpenCV gaze coordinates;
- `debug/step_1.log` and `debug/step_2.log`.

The final table columns are:

```text
user_id,slide_id,state_id,time,learning_element,element_type,duration
```

Example:

```text
R4,slide_01,slide_01_base,360.0,title,slide_title,20.0
R4,slide_01,slide_01_base,380.0,paragraph 1,paragraph,15.5
```
