#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PATH="$project_dir/.local/runtime/bin:$project_dir/.local/nemo-speech/bin:$PATH"

# Prefer OpenAI's standalone installer location even when this terminal has not
# reloaded its shell profile. The VS Code extension binary is fallback-only.
if [ -x "$HOME/.local/bin/codex" ]; then
  PATH="$HOME/.local/bin:$PATH"
elif ! command -v codex >/dev/null 2>&1; then
  codex_bin=
  for candidate in \
    "$HOME"/.vscode/extensions/openai.chatgpt-*/bin/macos-*/codex \
    "$HOME"/.vscode-insiders/extensions/openai.chatgpt-*/bin/macos-*/codex
  do
    if [ -x "$candidate" ]; then
      codex_bin=$candidate
    fi
  done
  if [ -n "$codex_bin" ]; then
    PATH="${codex_bin%/*}:$PATH"
  fi
fi

export PATH

cd "$project_dir"
exec "$project_dir/.local/runtime/bin/video-subtitle-pipeline" \
  --config "$project_dir/config.local.json" "$@"
