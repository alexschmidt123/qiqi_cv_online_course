#!/usr/bin/env bash
set -e

# Usage:
#   ./run.sh -dir <path> [-interval <seconds>]
#   path = single video file (.mp4/.mov) or directory containing videos
#   Example: bash run.sh -dir data/video/R10,P8_2.mp4
#   Example: bash run.sh -dir data/video -interval 2
#   Default -dir: data/video. Default -interval: 1 (one section per second).
# Results: output/<video_basename>/ = final_gaze_table.csv, full_text_combined.txt, validation/
# Requires conda env `cv_env` (see README); activates it when CONDA_DEFAULT_ENV is unset.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_DIR_INPUT="$ROOT/data/video"
INTERVAL_SEC="1"

repeat_char() {
  local char="$1"
  local count="$2"
  local out=""
  local i
  for (( i=0; i<count; i++ )); do
    out="${out}${char}"
  done
  printf "%s" "$out"
}

print_progress() {
  local cur="$1"
  local total="$2"
  local label="$3"
  local base="$4"
  local vid_idx="$5"
  local vid_total="$6"
  local width=34
  local pct=$(( cur * 100 / total ))
  local filled=$(( cur * width / total ))
  local empty=$(( width - filled ))
  local bar
  bar="$(repeat_char "=" "$filled")$(repeat_char "." "$empty")"
  printf "\r[%s] %3d%%  step %d/%d  %-13s  video %d/%d  %s" \
    "$bar" "$pct" "$cur" "$total" "$label" "$vid_idx" "$vid_total" "$base"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -dir)
      shift
      [ -n "${1:-}" ] || { echo "Usage: run.sh -dir <video_file_or_dir> [-interval <seconds>]" >&2; exit 1; }
      VIDEO_DIR_INPUT="$1"
      shift
      ;;
    -interval)
      shift
      [ -n "${1:-}" ] || { echo "Usage: run.sh -dir <path> [-interval <seconds>]" >&2; exit 1; }
      INTERVAL_SEC="$1"
      shift
      ;;
    *)
      echo "Usage: run.sh -dir <video_file_or_dir> [-interval <seconds>]" >&2
      exit 1
      ;;
  esac
done

# If -dir points to a file, use it as the only video and set report dir to its parent
VIDEOS_LIST=()
if [ -f "$VIDEO_DIR_INPUT" ]; then
  case "$VIDEO_DIR_INPUT" in
    *.mp4|*.mov) VIDEOS_LIST=("$VIDEO_DIR_INPUT"); VIDEO_DIR_INPUT="$(dirname "$VIDEO_DIR_INPUT")" ;;
    *) echo "Not a video file (.mp4/.mov): $VIDEO_DIR_INPUT" >&2; exit 1 ;;
  esac
fi

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate cv_env 2>/dev/null || echo "Warning: activate cv_env manually if needed." >&2
  fi
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
  outdir="$ROOT/output/$name"
  debugdir="$outdir/debug"
  echo
  echo "=== [$video_idx/$video_total] $base ==="
  mkdir -p "$outdir"
  mkdir -p "$debugdir"
  print_progress 1 4 "detect gaze" "$base" "$video_idx" "$video_total"

  python3 "$ROOT/script/step_1_detect_gaze.py" --video "$v" --output-dir "$outdir" > "$debugdir/step1.log" 2>&1

  print_progress 2 4 "detect section" "$base" "$video_idx" "$video_total"
  python3 "$ROOT/script/step_2_detect_section.py" \
    --video "$v" \
    --gaze-csv "$outdir/gaze_coordinates.csv" \
    --output-dir "$outdir" \
    --interval "$INTERVAL_SEC" > "$debugdir/step2.log" 2>&1

  # Optional GPT refinement: full_text_combined.txt + final_gaze_table.csv
  if [ -n "${OPENAI_API_KEY:-}" ] || [ -f "$ROOT/.env" ]; then
    print_progress 3 4 "gpt refine" "$base" "$video_idx" "$video_total"
    python3 "$ROOT/script/step_3_refine_with_gpt.py" --output-dir "$outdir" --interval "$INTERVAL_SEC" > "$debugdir/step3.log" 2>&1

    # Optional final visual validation: 3 key-frame images using final_gaze_table.csv
    if [ -f "$outdir/final_gaze_table.csv" ]; then
      print_progress 4 4 "validate" "$base" "$video_idx" "$video_total"
      python3 "$ROOT/script/step_4_validate_outputs.py" \
        --video "$v" \
        --final-csv "$outdir/final_gaze_table.csv" \
        --validation-dir "$outdir/validation" \
        --interval "$INTERVAL_SEC" > "$debugdir/step4.log" 2>&1

      # Keep only two files + validation folder for a neat output
      for f in gaze_coordinates.csv gaze_with_section.csv full_text.txt gaze_section_gpt_report.md gaze_check_report.txt gaze_ocr_log.txt gaze_coordinates_validation.txt; do
        [ -f "$outdir/$f" ] && rm -f "$outdir/$f"
      done
    fi
    print_progress 4 4 "done" "$base" "$video_idx" "$video_total"
    printf "\n"
  else
    print_progress 3 4 "skip gpt" "$base" "$video_idx" "$video_total"
    printf "\n"
    print_progress 4 4 "done" "$base" "$video_idx" "$video_total"
    printf "\n"
  fi
done

echo
echo "Done. Results under $ROOT/output/<video_name>/"
