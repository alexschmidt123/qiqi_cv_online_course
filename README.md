# Online-course attention analysis

## Introduction

This project analyzes where a user looks while viewing an online course.

- Article videos: OpenCV detects gaze, OCR reads words around it, and NLP matches
  those words to the reconstructed full article and its sections.
- Slide videos: OpenCV matches slides and popup states to one frozen shared
decision library. Every output decision must be selected from that library;
  an unmatched state stops the run instead of creating a new decision.

Article output contains `attention_table.csv`, `full_article.txt`, and sampled
decision screenshots without section borders. Slide output contains the CSV and
one validation screenshot per distinct slide or popup state. The prepared,
CSV-only slide library is stored in `data/slide_standard_library/`.

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

For slide runs, `run.sh` automatically prepares the shared library when it is
missing or incomplete and skips preparation when all 22+ slide entries and
their files pass validation. Force a rebuild with:

```bash
bash run.sh -dir data/video_slide/R4_P1.mp4 -prepare-slide-library -model gpt-4o
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
