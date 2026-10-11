#!/usr/bin/env bash
set -euo pipefail

SPEED=0.25
DRY_RUN=false
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed) SPEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) INPUTS+=("$1"); shift ;;
  esac
done

[[ ${#INPUTS[@]} -gt 0 ]] || { echo "usage: $0 [--speed 0.25] VIDEO_OR_DIRECTORY..." >&2; exit 2; }
awk -v speed="$SPEED" 'BEGIN { exit !(speed > 0 && speed <= 1) }' || {
  echo "speed must be greater than 0 and at most 1" >&2
  exit 2
}
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 2; }

VIDEOS=()
for input in "${INPUTS[@]}"; do
  if [[ -d "$input" ]]; then
    while IFS= read -r -d '' video; do
      [[ "$video" == *_slow_*x.mp4 ]] || VIDEOS+=("$video")
    done < <(find "$input" -type f -name '*.mp4' -print0 | sort -z)
  elif [[ -f "$input" && "$input" == *.mp4 ]]; then
    VIDEOS+=("$input")
  else
    echo "not an MP4 file or directory: $input" >&2
    exit 2
  fi
done
[[ ${#VIDEOS[@]} -gt 0 ]] || { echo "no MP4 videos found" >&2; exit 2; }

for source in "${VIDEOS[@]}"; do
  stem="${source%.mp4}"
  output="${stem}_slow_${SPEED}x.mp4"
  command=(ffmpeg -n -loglevel error -i "$source" -an -vf "setpts=PTS/$SPEED" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "$output")
  if $DRY_RUN; then
    printf '%q ' "${command[@]}"
    printf '\n'
  elif [[ -e "$output" ]]; then
    echo "skip existing: $output"
  else
    "${command[@]}"
    echo "saved: $output"
  fi
done
