#!/usr/bin/env bash
set -euo pipefail

# Unified online-course attention pipeline.
# Step 1: pure OpenCV gaze coordinates.
# Step 2: OCR evidence + AI video/layout analysis and report.
# Step 3: deterministic gaze-to-element dwell events.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_INPUT="$ROOT/data/video"
VIDEO_TYPE="auto"
MODEL="gpt-5.6-sol"
SAMPLE_INTERVAL="2"
MIN_DURATION="0.1"

usage() {
  echo "Usage: bash run.sh -dir <video_or_directory> [-type auto|article|slides] [-model model] [-sample-interval sec] [-min-duration sec]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -dir) shift; [ -n "${1:-}" ] || { usage; exit 1; }; VIDEO_INPUT="$1"; shift ;;
    -type) shift; [ -n "${1:-}" ] || { usage; exit 1; }; VIDEO_TYPE="$1"; shift ;;
    -model) shift; [ -n "${1:-}" ] || { usage; exit 1; }; MODEL="$1"; shift ;;
    -sample-interval) shift; [ -n "${1:-}" ] || { usage; exit 1; }; SAMPLE_INTERVAL="$1"; shift ;;
    -min-duration) shift; [ -n "${1:-}" ] || { usage; exit 1; }; MIN_DURATION="$1"; shift ;;
    *) usage; exit 1 ;;
  esac
done

case "$VIDEO_TYPE" in auto|article|slides) ;; *) usage; exit 1 ;; esac
case "$VIDEO_INPUT" in /*) ;; *) VIDEO_INPUT="$ROOT/$VIDEO_INPUT" ;; esac

videos=()
if [ -f "$VIDEO_INPUT" ]; then
  videos=("$VIDEO_INPUT")
elif [ -d "$VIDEO_INPUT" ]; then
  while IFS= read -r -d '' video; do videos+=("$video"); done < <(
    find "$VIDEO_INPUT" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' \) -print0 | sort -z
  )
else
  echo "Video path not found: $VIDEO_INPUT" >&2; exit 1
fi
[ "${#videos[@]}" -gt 0 ] || { echo "No videos found" >&2; exit 1; }

if [ -z "${CONDA_DEFAULT_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate cv_env 2>/dev/null || true
fi

for video in "${videos[@]}"; do
  filename="$(basename "$video")"
  video_id="${filename%.*}"
  user_id="${video_id%%,*}"; user_id="${user_id%%_*}"
  outdir="$ROOT/output/$video_id"
  mkdir -p "$outdir/debug"

  echo "[$video_id] 1/3 Pure OpenCV gaze detection"
  python3 "$ROOT/script/step_1_detect_gaze.py" --video "$video" --output-dir "$outdir" \
    > "$outdir/debug/step_1.log" 2>&1

  echo "[$video_id] 2/3 OCR + AI layout analysis and report"
  python3 "$ROOT/script/step_2_analyze_video_with_ai.py" --video "$video" --output-dir "$outdir" \
    --type "$VIDEO_TYPE" --model "$MODEL" --sample-interval "$SAMPLE_INTERVAL" \
    > "$outdir/debug/step_2.log" 2>&1

  echo "[$video_id] 3/3 Attention event table"
  python3 "$ROOT/script/step_3_map_gaze_to_events.py" \
    --library "$outdir/element_library.json" --gaze-csv "$outdir/gaze_coordinates.csv" \
    --user-id "$user_id" --min-duration "$MIN_DURATION" --output "$outdir/attention_table.csv"

  echo "[$video_id] Complete: attention_table.csv, ai_report.md, ai_report.json, layout/*.png"
done
