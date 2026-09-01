#!/bin/zsh
set -eu

project_dir=${0:A:h}
source_app="$project_dir/macos/Video Subtitle Pipeline.app"
desktop_app="$HOME/Desktop/Video Subtitle Pipeline.app"
output_dir="$HOME/Movies/Video Subtitle Pipeline Outputs"
zshrc="$HOME/.zshrc"

mkdir -p "$output_dir"
mkdir -p "$desktop_app/Contents"
cp -R "$source_app/Contents/" "$desktop_app/Contents/"
mkdir -p "$desktop_app/Contents/MacOS"
mkdir -p "$desktop_app/Contents/Resources"
print -r -- "$project_dir" >| "$desktop_app/Contents/Resources/project-path"
xcrun clang -fobjc-arc -fblocks -fmodules-cache-path="$project_dir/.local/clang-module-cache" \
  -O2 -framework Cocoa \
  "$project_dir/macos/VideoSubtitleProgress.m" \
  -o "$desktop_app/Contents/MacOS/video-subtitle-droplet"
codesign --force --deep --sign - "$desktop_app" >/dev/null
touch "$desktop_app"

begin_marker="# >>> video-subtitle-pipeline >>>"
end_marker="# <<< video-subtitle-pipeline <<<"
alias_line="alias videosubs='$project_dir/run-local-drop.sh'"

touch "$zshrc"
if grep -Fq "$begin_marker" "$zshrc" && grep -Fq "$end_marker" "$zshrc"; then
  temp_file=$(mktemp "$zshrc.tmp.XXXXXX")
  awk -v begin="$begin_marker" -v end="$end_marker" -v alias_line="$alias_line" '
    $0 == begin {
      print begin
      print alias_line
      replacing = 1
      next
    }
    replacing && $0 == end {
      print end
      replacing = 0
      next
    }
    !replacing { print }
  ' "$zshrc" >| "$temp_file"
  chmod "$(stat -f '%Lp' "$zshrc")" "$temp_file"
  mv "$temp_file" "$zshrc"
else
  {
    print
    print -r -- "$begin_marker"
    print -r -- "$alias_line"
    print -r -- "$end_marker"
  } >> "$zshrc"
fi

print "Installed Desktop app: $desktop_app"
print "Output folder: $output_dir"
print "Installed zsh alias: videosubs"
print "Reload it now with: source \"$zshrc\""
