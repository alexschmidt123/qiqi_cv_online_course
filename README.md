# Online-course attention analysis

## Introduction

This project analyzes where a user looks while viewing an online course.

- Article videos: OpenCV detects gaze, OCR reads text around the gaze, and AI
  identifies the article text, layout, and section being viewed.
- Slide videos: OpenCV finds unique slide and popup patterns, and AI identifies
  their titles, paragraphs, figures, buttons, popups, and website elements.

Each video produces one `attention_table.csv` report and sampled validation
screenshots showing layout borders, the gaze point, the selected element, and the
decision explanation.

## Installation

```bash
conda create -n cv_env python=3.11 -y
conda activate cv_env
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

The default model is `gpt-4o-mini`, which supports screenshot inputs and keeps
the AI analysis cost relatively low.

## Run

Process all article videos:

```bash
bash run.sh -dir data/video_article
```

Process one article video:

```bash
bash run.sh -dir data/video_article/R4,P1_1.mp4
```

Process all slide videos:

```bash
bash run.sh -dir data/video_slide
```

Process one slide video:

```bash
bash run.sh -dir data/video_slide/R4_P1.mp4
```

Results are written to:

```text
output/article_<video-name>/
output/slide_<video-name>/
```
