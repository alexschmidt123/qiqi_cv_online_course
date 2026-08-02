#!/usr/bin/env bash
set -euo pipefail

# Unified online-course attention pipeline.
# Step 1: pure OpenCV gaze coordinates.
# Step 2: OCR evidence + AI video/layout analysis and report.
# Step 3: deterministic gaze-to-element dwell events.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_INPUT="$ROOT/data/video_article"
MODEL="gpt-4o-mini"
SAMPLE_INTERVAL="1"
AI_SCREENSHOTS="6"
MIN_DURATION="0.1"

usage() {
  echo "Usage: bash run.sh -dir <video_article|video_slide directory or a video inside one> [-model model] [-sample-interval sec] [-ai-screenshots count] [-min-duration sec]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -dir) shift; [ -n "${1:-}" ] || { usage; exit 1; }; VIDEO_INPUT="$1"; shift ;;
    -model) shift; [ -n "${1:-}" ] || { usage; exit 1; }; MODEL="$1"; shift ;;
    -sample-interval) shift; [ -n "${1:-}" ] || { usage; exit 1; }; SAMPLE_INTERVAL="$1"; shift ;;
    -ai-screenshots) shift; [ -n "${1:-}" ] || { usage; exit 1; }; AI_SCREENSHOTS="$1"; shift ;;
    -min-duration) shift; [ -n "${1:-}" ] || { usage; exit 1; }; MIN_DURATION="$1"; shift ;;
    *) usage; exit 1 ;;
  esac
done

case "$VIDEO_INPUT" in /*) ;; *) VIDEO_INPUT="$ROOT/$VIDEO_INPUT" ;; esac

# Infer type strictly from the input directory. A single file uses its parent.
type_dir="$VIDEO_INPUT"
[ -f "$type_dir" ] && type_dir="$(dirname "$type_dir")"
type_folder="$(basename "$type_dir")"
VIDEO_TYPE_LABEL="${type_folder##*_}"
case "$type_folder" in
  video_article) VIDEO_TYPE="article" ;;
  video_slide) VIDEO_TYPE="slides" ;;
  *)
    echo "Directory must be named video_article or video_slide: $type_dir" >&2
    exit 1
    ;;
esac
echo "Video type: $VIDEO_TYPE (input: $VIDEO_INPUT)"

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
  outdir="$ROOT/output/${VIDEO_TYPE_LABEL}_${video_id}"
  analysis_dir="$outdir/debug/analysis"
  mkdir -p "$analysis_dir"

  echo "[$video_id] 1/4 Pure OpenCV gaze detection"
  python3 "$ROOT/script/step_1_detect_gaze.py" --video "$video" --output-dir "$outdir/debug" \
    --output-name "gaze_coordinates.internal" --no-validation \
    > "$outdir/debug/step_1.log" 2>&1

  echo "[$video_id] 2/4 Type-specific AI layout analysis"
  python3 "$ROOT/script/step_2_analyze_video_with_ai.py" --video "$video" --output-dir "$analysis_dir" \
    --gaze-csv "$outdir/debug/gaze_coordinates.internal" \
    --type "$VIDEO_TYPE" --model "$MODEL" --sample-interval "$SAMPLE_INTERVAL" \
    --ai-screenshots "$AI_SCREENSHOTS" \
    > "$outdir/debug/step_2.log" 2>&1

  echo "[$video_id] 3/4 Chronological gaze and duration tables"
  python3 "$ROOT/script/step_3_map_gaze_to_events.py" \
    --library "$analysis_dir/element_library.json" --gaze-csv "$outdir/debug/gaze_coordinates.internal" \
    --user-id "$user_id" --min-duration "$MIN_DURATION" \
    --output "$outdir/attention_table.csv"

  echo "[$video_id] 4/4 Visual validation"
  python3 "$ROOT/script/step_4_validate_attention.py" --video "$video" \
    --library "$analysis_dir/element_library.json" \
    --gaze-data "$outdir/debug/gaze_coordinates.internal" \
    --output-dir "$outdir/validation" > "$outdir/debug/step_4.log" 2>&1

  echo "[$video_id] Complete: attention_table.csv and validation/*.png"
done
