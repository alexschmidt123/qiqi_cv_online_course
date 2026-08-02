# Online-course attention analysis

## Introduction

This project analyzes where a user looks while viewing an online course.

- Article videos: OpenCV detects gaze, OCR reads words around it, and NLP matches
  those words to the reconstructed full article and its sections.
- Slide videos: OpenCV matches slides and popup states to one shared course
  library; AI identifies elements only when a previously unseen state appears.

Article output contains `attention_table.csv`, `full_article.txt`, and sampled
decision screenshots without section borders. Slide output contains the CSV and
one validation screenshot per distinct slide or popup state. The reusable slide
library is stored in `output/slide_standard_library/`.

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
