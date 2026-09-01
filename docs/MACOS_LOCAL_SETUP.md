# Fully local ASR setup on Apple Silicon

This setup was validated end to end on 2026-09-01 using a 10-core Apple M5
MacBook Pro with 32 GB unified memory and macOS 26.5.1. The tested workflow used
Metal-accelerated NeMo-Speech.cpp for whole-file Nemotron 3.5 ASR, a
repository-local Python/FFmpeg environment, Codex CLI with GPT-5.6 Luna/high for
Korean translation, and Apple SD Gothic Neo for hard subtitles.

The validation included a 148-second 1920x1080 iPhone recording. Whole-file ASR
returned native word timestamps, all nonempty IDs were translated, subtitle
times did not overlap, and the Korean hard-sub output was visually inspected.
This is a successful compatibility result, not a general performance or word
error rate benchmark.

## 1. Apple developer tools

NeMo-Speech.cpp's Metal build requires Apple Silicon and the Xcode Command Line
Tools:

```bash
xcode-select --install
xcode-select -p
```

The validated machine did not require Homebrew or administrator access. All
runtime files below live in `.local/` or `models/`, both ignored by Git.

## 2. Repository-local Python and FFmpeg

Download the ARM64 micromamba archive and create a conda-forge-only environment:

```bash
curl -fL -o /tmp/micromamba-osx-arm64.tar.bz2 \
  https://micro.mamba.pm/api/micromamba/osx-arm64/latest
mkdir -p .local/micromamba-extract
tar -xjf /tmp/micromamba-osx-arm64.tar.bz2 -C .local/micromamba-extract

MAMBA_ROOT_PREFIX="$PWD/.local/mamba-root" \
  .local/micromamba-extract/bin/micromamba create -y \
  -p "$PWD/.local/runtime" --override-channels -c conda-forge \
  python=3.13 ffmpeg

.local/runtime/bin/python -m pip install -e .
```

The tested conda-forge FFmpeg build included `ffprobe`, libass, fontconfig,
HarfBuzz, H.264, and AAC support. Confirm the important pieces:

```bash
.local/runtime/bin/python --version
.local/runtime/bin/ffmpeg -version
.local/runtime/bin/ffmpeg -filters | grep subtitles
```

## 3. NeMo-Speech.cpp with Metal

Download and inspect NVIDIA's installer, then install the published Apple
Silicon Metal artifact into this repository:

```bash
curl -fsSLo /tmp/nemo-speech-install.sh \
  https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/scripts/install.sh

/bin/sh /tmp/nemo-speech-install.sh \
  --prefix "$PWD/.local/nemo-speech" \
  --backend metal \
  --binary-only \
  --no-modify-path

.local/nemo-speech/bin/nemo-speech --version
```

Version 0.1.0 was the published Metal artifact used during validation. Check
NVIDIA's current release and installer before assuming that version remains
current.

## 4. Nemotron 3.5 Q8 model

The official Q8 GGUF is the compact model format used by the Metal runtime:

```bash
mkdir -p models
curl -fL --retry 3 \
  -o models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/main/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf

shasum -a 256 models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
```

The validated artifact was 741,548,352 bytes with SHA-256:

```text
a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae
```

That digest corresponds to the tested Hugging Face revision. Re-check the
official repository metadata if the upstream artifact changes.

## 5. Standalone Codex CLI

Codex is not used for ASR. It receives transcript text for translation and, in
segmented compatibility mode, stitching. Install the standalone CLI rather
than depending on an IDE-bundled executable:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
codex login
codex login status
```

The Desktop launcher prefers `~/.local/bin/codex`, then `PATH`, and only then
falls back to the OpenAI VS Code extension binary. Never commit or copy
`~/.codex/auth.json`.

## 6. Local configuration

Copy the tested relative-path template. The destination is ignored by Git:

```bash
cp config.macos.example.json config.local.json
./run-local.sh /path/to/video.mp4 --dry-run
```

The template selects whole-file local Metal ASR, Luna/high Korean translation,
bilingual SRT, and hard-sub rendering. Change `asr_language` when the spoken
language is known and is not US English.

Run the project checks before processing important media:

```bash
PATH="$PWD/.local/runtime/bin:$PATH" PYTHONPATH=src \
  .local/runtime/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .local/runtime/bin/python -m compileall -q src providers tests
```

## 7. Desktop droplet and terminal alias

Install the native drag-and-drop app and `videosubs` zsh alias:

```bash
./install-macos-shortcuts.sh
source "$HOME/.zshrc"
```

Dropping files or a folder onto **Video Subtitle Pipeline.app** prompts for an
output directory, defaulting to `~/Movies/Video Subtitle Pipeline Outputs`.
The native window reports per-video phases and shows a final success/error
dialog with output paths. Successful runs can play the rendered video or show
it in Finder; failed runs can open the log or output folder. Folder inputs are
searched recursively. The source folder itself is a valid destination; generated
`*.ko-bilingual.hardsub.*` files are skipped to avoid processing them again.

The equivalent terminal commands are:

```bash
videosubs "$HOME/Downloads/video.mp4"
videosubs "$HOME/Downloads/folder of videos"
```

Set `VIDEO_SUBTITLE_OUTPUT_DIR` to override the alias destination.

## Privacy boundary

- Local command ASR receives the extracted mono WAV and does not upload audio.
- Codex translation receives transcript text, IDs, and timing context—not the
  video or audio.
- Frames are not sent unless visual review is configured and
  `--allow-frame-upload` is explicitly enabled.
- FFmpeg may copy global source metadata, including iPhone location tags, into
  a rendered output. Inspect or strip metadata before sharing media externally.

See [Provider setup](PROVIDERS.md) and [Known limitations](LIMITATIONS.md) for
provider alternatives and quality caveats.
