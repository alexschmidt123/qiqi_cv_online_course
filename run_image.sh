#!/usr/bin/env bash
set -e
set -o pipefail

# PPT / image-style videos: gaze + element detection + optional ChatGPT refinement.
# Requires conda env `cv_env` (see README); activates it when CONDA_DEFAULT_ENV is unset.
# Usage:
#   bash run_image.sh -dir data/video_image/R10_P8.mp4
#   bash run_image.sh -dir data/video_image -interval 0.5
#   bash run_image.sh -test-image path/to/screenshot.png   # detectors only; no video pipeline
#   bash run_image.sh -test-folder data/images           # batch: output_image/test_result/<stem>/
# Default -dir: data/video_image. Default -interval: 0.5 (seconds per element-detection bucket).
# With OPENAI_API_KEY or .env: step 2 uses vision API for PPT rectangle (once); step 3 GPT refine; validation.
# Outputs: output_image/<video_basename>/final_gaze_table.txt, validation/ (CSVs under debug/)
# Single-image test: output_image/<png_stem>/original.png + labeled.png only.
# Folder test: output_image/test_result/<stem>/original.png + labeled.png per image.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "-test-image" ]; then
  shift
  [ -n "${1:-}" ] || { echo "Usage: bash run_image.sh -test-image <path/to.png|jpg>" >&2; exit 1; }
  IMG="$1"
  case "$IMG" in
    /*) ;;
    *) IMG="$ROOT/$IMG" ;;
  esac
  if [ ! -f "$IMG" ]; then
    echo "File not found: $IMG" >&2
    exit 1
  fi
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)" 2>/dev/null || true
      conda activate cv_env 2>/dev/null || echo "Warning: activate cv_env manually if needed." >&2
    fi
  fi
  STEM=$(basename "$IMG")
  STEM="${STEM%.*}"
  echo "=== Single-image overlay test: $IMG ==="
  python3 "$ROOT/scripts_image/test_single_image_overlay.py" --image "$IMG" --output-root "$ROOT/output_image"
  echo "Done -> $ROOT/output_image/$STEM/ (original.png, labeled.png)"
  exit 0
fi

if [ "${1:-}" = "-test-folder" ]; then
  shift
  [ -n "${1:-}" ] || { echo "Usage: bash run_image.sh -test-folder <path/to/folder>" >&2; exit 1; }
  FOLDER="$1"
  case "$FOLDER" in
    /*) ;;
    *) FOLDER="$ROOT/$FOLDER" ;;
  esac
  if [ ! -d "$FOLDER" ]; then
    echo "Not a directory: $FOLDER" >&2
    exit 1
  fi
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)" 2>/dev/null || true
      conda activate cv_env 2>/dev/null || echo "Warning: activate cv_env manually if needed." >&2
    fi
  fi
  echo "=== Single-image batch: $FOLDER -> output_image/test_result/ ==="
  python3 "$ROOT/scripts_image/test_single_image_overlay.py" --folder "$FOLDER" --output-root "$ROOT/output_image"
  echo "Done -> $ROOT/output_image/test_result/<image_stem>/ (original.png, labeled.png each)"
  exit 0
fi

VIDEO_DIR_INPUT="$ROOT/data/video_image"
INTERVAL_SEC="0.5"
THRESHOLD_PX="18"

while [ $# -gt 0 ]; do
  case "$1" in
    -dir)
      shift
      [ -n "${1:-}" ] || { echo "Usage: bash run_image.sh -dir <video_or_dir> [-interval sec] [-threshold px]" >&2; exit 1; }
      VIDEO_DIR_INPUT="$1"
      shift
      ;;
    -interval)
      shift
      [ -n "${1:-}" ] || { echo "Usage: bash run_image.sh -dir <path> [-interval sec] [-threshold px]" >&2; exit 1; }
      INTERVAL_SEC="$1"
      shift
      ;;
    -threshold)
      shift
      [ -n "${1:-}" ] || { echo "Usage: bash run_image.sh -dir <path> [-interval sec] [-threshold px]" >&2; exit 1; }
      THRESHOLD_PX="$1"
      shift
      ;;
    *)
      echo "Usage: bash run_image.sh -dir <video_file_or_dir> [-interval seconds] [-threshold px]" >&2
      exit 1
      ;;
  esac
done

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate cv_env 2>/dev/null || echo "Warning: activate cv_env manually if needed." >&2
  fi
fi

VIDEOS_LIST=()
if [ -f "$VIDEO_DIR_INPUT" ]; then
  case "$VIDEO_DIR_INPUT" in
    *.mp4|*.mov) VIDEOS_LIST=("$VIDEO_DIR_INPUT"); VIDEO_DIR_INPUT="$(dirname "$VIDEO_DIR_INPUT")" ;;
    *) echo "Not a video file (.mp4/.mov): $VIDEO_DIR_INPUT" >&2; exit 1 ;;
  esac
fi

echo "Using video path: $VIDEO_DIR_INPUT"

if [ "${#VIDEOS_LIST[@]}" -eq 0 ]; then
  shopt -s nullglob
  videos=("$VIDEO_DIR_INPUT"/*.mp4 "$VIDEO_DIR_INPUT"/*.mov)
  shopt -u nullglob
else
  videos=("${VIDEOS_LIST[@]}")
fi

if [ "${#videos[@]}" -eq 0 ]; then
  echo "No .mp4 or .mov files found."
  exit 1
fi

video_total="${#videos[@]}"
video_idx=0
for v in "${videos[@]}"; do
  video_idx=$((video_idx + 1))
  base="$(basename "$v")"
  name="${base%.*}"
  outdir="$ROOT/output_image/$name"
  debugdir="$outdir/debug"
  echo
  echo "=== [$video_idx/$video_total] PPT pipeline: $base ==="
  mkdir -p "$outdir"
  mkdir -p "$debugdir"

  echo "  [1/4] detect gaze..."
  python3 "$ROOT/scripts_image/step_1_detect_gaze.py" --video "$v" --output-dir "$debugdir" 2>&1 | tee "$debugdir/step1.log"

  echo "  [2/4] detect elements + preliminary final_gaze_table..."
  python3 "$ROOT/scripts_image/step_2_detect_elements.py" \
    --video "$v" \
    --gaze-csv "$debugdir/gaze_coordinates.csv" \
    --output-dir "$outdir" \
    --interval "$INTERVAL_SEC" \
    --threshold "$THRESHOLD_PX" 2>&1 | tee "$debugdir/step2.log"

  if [ -n "${OPENAI_API_KEY:-}" ] || [ -f "$ROOT/.env" ]; then
    echo "  [3/4] GPT refine element names..."
    python3 "$ROOT/scripts_image/step_3_refine_elements_gpt.py" \
      --output-dir "$outdir" \
      --video "$v" \
      --interval "$INTERVAL_SEC" 2>&1 | tee "$debugdir/step3.log" || {
      echo "  Warning: step 3 failed; keeping step-2 final_gaze_table.csv" >&2
    }
  else
    echo "  [3/4] skip GPT (set OPENAI_API_KEY or add .env with OPENAI_API_KEY)"
  fi

  if [ -f "$outdir/final_gaze_table.txt" ] || [ -f "$debugdir/final_gaze_table.csv" ]; then
    echo "  [4/4] validation images..."
    python3 "$ROOT/scripts_image/step_4_validate_image.py" \
      --video "$v" \
      --final-csv "$outdir/final_gaze_table.txt" \
      --output-dir "$outdir" 2>&1 | tee "$debugdir/step4.log"
  fi
  echo "  Done -> $outdir (final_gaze_table.txt + validation/; intermediates in debug/)"
done

echo
echo "Finished. Deliverables: output_image/<video_name>/final_gaze_table.txt and validation/"
