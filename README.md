# Gaze-to-Section Pipeline

> New architecture: for reusable slide/webpage element polygons and the final
> `time, learning_element, duration` event output, see
> [`script_v2/README.md`](script_v2/README.md). It avoids per-time-bucket AI calls.

## 1) Install

Use conda only. Create and use an env named **`cv_env`** (the run scripts activate it in bash when it is not already active).

```bash
conda create -n cv_env python=3.11 -y
conda activate cv_env
cd /path/to/qiqi_cv_online_course
python -m pip install -r requirements.txt
```

`cv2` is from `opencv-python`, installed into that env.

Optional (for GPT step):

```bash
export OPENAI_API_KEY="your_key_here"
```

## 2) Run

Default interval is `1` second:

```bash
bash run.sh -dir data/video
bash run.sh -dir data/video/R4,P1_1.mp4
```

Custom interval:

```bash
bash run.sh -dir data/video -interval 2
bash run.sh -dir data/video/R4,P1_1.mp4 -interval 0.5
```

## 3) PPT / image-style (`run_image.sh`)

Defaults: `-dir data/video_image`, `-interval 0.5`, `-threshold 18`. Optional GPT if `OPENAI_API_KEY` or `.env` is set.

```bash
bash run_image.sh -dir data/video_image              # or a single .mp4/.mov
bash run_image.sh -dir data/video_image/foo.mp4 -interval 1 -threshold 18
```

Output: `output_image/<name>/final_gaze_table.txt`, `validation/`, `debug/`.
