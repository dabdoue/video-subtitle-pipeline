#!/bin/zsh
set -u

project_dir=${0:A:h}
output_dir=${VIDEO_SUBTITLE_OUTPUT_DIR:-"$HOME/Movies/Video Subtitle Pipeline Outputs"}
mkdir -p "$output_dir"
output_dir=${output_dir:A}

if (( $# == 0 )); then
  print -u2 "usage: ${0:t} VIDEO_OR_FOLDER [...]"
  print -u2 "outputs: $output_dir"
  exit 2
fi

typeset -a videos
typeset -A discovered

add_video() {
  local video=${1:A}
  local scan_root=${2:-}
  local extension=${video:e:l}
  [[ -f "$video" ]] || return
  # When the selected folder contains a separate output subfolder, do not
  # recursively process previous results. If the user deliberately chooses the
  # selected folder itself (or one of its parents) as the destination, keep the
  # source videos eligible and only ignore our own generated hard-sub files.
  if [[ -n "$scan_root" && "$output_dir" == "$scan_root"/* && "$video" == "$output_dir"/* ]]; then
    return
  fi
  if [[ -n "$scan_root" && "$video" == "$output_dir"/* && "${video:t}" == *.ko-bilingual.hardsub.* ]]; then
    return
  fi
  case "$extension" in
    mp4|mov|mkv|m4v|webm) ;;
    *) return ;;
  esac
  if [[ -z ${discovered[$video]-} ]]; then
    videos+=("$video")
    discovered[$video]=1
  fi
}

for input in "$@"; do
  if [[ -d "$input" ]]; then
    scan_root=${input:A}
    while IFS= read -r -d $'\0' video; do
      add_video "$video" "$scan_root"
    done < <(
      find "$input" -type f \( \
        -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o \
        -iname '*.m4v' -o -iname '*.webm' \
      \) -print0
    )
  elif [[ -f "$input" ]]; then
    # An explicitly dropped file is always eligible, including when the user
    # chooses its current directory as the output destination.
    add_video "$input"
  else
    print -u2 "warning: skipping missing input: $input"
  fi
done

if (( ${#videos} == 0 )); then
  print -u2 "error: no supported videos found"
  exit 1
fi

typeset -A stem_counts
integer completed=0
integer failed=0
integer index=0

for video in "${videos[@]}"; do
  (( index += 1 ))
  stem=${video:t:r}
  count=${stem_counts[$stem]:-0}
  (( count += 1 ))
  stem_counts[$stem]=$count
  if (( count > 1 )); then
    stem="$stem-$count"
  fi

  print "[$index/${#videos}] Processing: $video"
  if "$project_dir/run-local.sh" "$video" \
    --output-srt "$output_dir/$stem.ko-bilingual.srt" \
    --output-manifest "$output_dir/$stem.ko-bilingual.segments.json" \
    --output-video "$output_dir/$stem.ko-bilingual.hardsub.mp4"
  then
    (( completed += 1 ))
  else
    (( failed += 1 ))
    print -u2 "error: failed: $video"
  fi
done

print "Completed: $completed; failed: $failed"
print "Outputs: $output_dir"
(( failed == 0 ))
