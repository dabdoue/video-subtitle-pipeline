# Video Subtitle Pipeline

A repeatable, auditable workflow for turning a video into timed source
transcripts, translated subtitles, and an optional hard-subtitled MP4.

The default workflow sends one mono audio track to a timestamp-capable ASR
provider. Native word timestamps assign the continuous transcript into
five-second subtitle anchors without cutting model context at every boundary.
The bundled Nemotron server can choose fast offline encoding when the input
fits a configurable GPU-memory budget, or bounded cache-aware streaming for
longer inputs. The original independent-window workflow remains available with
`--asr-mode segmented`; it adds overlap and stitches boundary text.

## What it produces

For `training.mp4` and Korean output (`--language-code ko`):

```text
training.ko.srt
training.ko.segments.json
training.ko.hardsub.mp4   # only with --burn
```

The JSON manifest is the source of truth. It retains:

- nominal subtitle ranges;
- the ASR mode and audio range;
- raw source text and native word timestamps;
- optional per-word decoder confidence and calculation metadata;
- the source text assigned to each nominal anchor;
- frame-assisted review proposals, frame timestamps/hashes, and apply status;
- translation parts and rendered cue timing;
- provider/model names without credentials or the endpoint URL.

That separation makes errors reviewable and lets captions be rerendered from a
corrected manifest without guessing what happened at a boundary.

## Install

Requirements:

- Python 3.11 or newer;
- `ffmpeg` and `ffprobe`;
- `curl` for an OpenAI-compatible ASR endpoint;
- the standalone `codex` CLI for Codex stitching/translation;
- a target-language font, such as Noto CJK, for hard subtitles.

Install the CLI in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
video-subtitle-pipeline --help
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

For the tested repository-local Apple Silicon/Metal installation, including
NeMo-Speech.cpp, the Q8 model, FFmpeg, and the macOS droplet, follow
[Fully local ASR setup on Apple Silicon](docs/MACOS_LOCAL_SETUP.md).

### Codex CLI authentication

When `codex` is selected for stitching or translation, install the standalone
Codex CLI using OpenAI's current macOS/Linux installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
codex --version
```

Authenticate once with the user account that should authorize the requests,
then verify the cached session:

```bash
codex login
codex login status
```

The browser flow signs in with ChatGPT. API-key authentication is also
supported; follow the official authentication documentation rather than
placing a key in this repository. Never commit or copy `~/.codex/auth.json`.
The CLI and IDE extension share cached login details.

`run-local.sh` prefers the standalone CLI at `~/.local/bin/codex`, then an
existing `codex` on `PATH`. As a convenience fallback only, it can discover the
Codex binary bundled with the OpenAI VS Code extension. A fresh setup should
still install and authenticate the standalone CLI.

- Codex CLI: https://developers.openai.com/codex/cli
- Authentication: https://developers.openai.com/codex/auth

## Configuration

Copy the safe example and fill in settings for your machine:

```bash
cp config.example.json config.local.json
```

`config.local.json` and `.env` are ignored by Git. This checkout already has an
ignored local config matching the workflow used during development. Never move
private hosts, tokens, or personal paths into `config.example.json`.

Run with an explicit config:

```bash
video-subtitle-pipeline --config config.local.json /path/to/video.mp4
```

When run from this repository root, `config.local.json` is loaded automatically
if it exists. `VIDEO_SUBTITLE_CONFIG=/path/to/config.json` is also supported.
CLI arguments have highest precedence:

```text
CLI flag > JSON config > environment/default
```

Use `--no-burn`, `--no-include-source`, or similar Boolean negations to override
a `true` config value.

Before an expensive run:

```bash
video-subtitle-pipeline --config config.local.json video.mp4 --dry-run
```

### macOS drag-and-drop and batch launcher

Install the Desktop droplet and `videosubs` zsh alias:

```bash
./install-macos-shortcuts.sh
source "$HOME/.zshrc"
```

Drag a supported video or a folder onto **Video Subtitle Pipeline.app** on the
Desktop. Folders are searched recursively for MP4, MOV, MKV, M4V, and WebM
files. The app asks whether to use the default destination or choose another
folder, then shows per-video phase progress. Dismissing the folder picker falls
back to the default destination; the explicit Cancel button stops the run.
The destination may be the source folder itself; generated
`*.ko-bilingual.hardsub.*` files are excluded from recursive runs so they are
not processed again.

At completion, a success or error dialog shows the output directory and created
file paths. A successful run can immediately play the rendered video or show it
in Finder. A failed run can open its log or output folder. By default, results
and `latest-run.log` are written to:

```text
~/Movies/Video Subtitle Pipeline Outputs
```

Dropping another item while a run is active shows its current input, phase,
process ID, and output directory. From that dialog you can return to the
progress window, open the destination, or confirm that the current pipeline and
its child processes should be stopped. Logs and already-written files remain
after cancellation; an incomplete video may not be playable.

The terminal alias accepts one or more files and folders and uses the same
output directory:

```bash
videosubs "$HOME/Downloads/video.mp4"
videosubs "$HOME/Downloads/folder of videos"
```

Set `VIDEO_SUBTITLE_OUTPUT_DIR` to override the destination for either entry
point.

## Whole-file default and segmented fallback

With `--asr-mode whole`, FFmpeg extracts one mono 16 kHz audio track. The ASR
runtime processes it as one stateful stream and returns word timestamps. A word
is assigned to the five-second anchor containing its temporal midpoint. No LLM
stitching is needed, and the model retains context across subtitle boundaries.

Use `--asr-mode segmented` when a provider cannot return word timestamps or an
upload must be divided into bounded requests. In that mode, five-second anchors
and `--audio-overlap 1` produce the following windows:

With five-second anchors and `--audio-overlap 1`, nominal segment `5–10s` is
transcribed from audio `4–11s`. Its neighbors are `0–6s` and `9–16s`.

```text
nominal anchors:  [ 0-----5 ][ 5----10 ][10----15 ]
ASR windows:      [ 0--------6 ]
                         [ 4--------11 ]
                                  [ 9--------16 ]
```

The segmented-mode stitcher sees raw text from neighboring windows and returns one clean
transcript for each nominal anchor. It must assign overlap speech exactly once,
repair a cut word only when the neighboring result supports it, and preserve
uncertainty rather than inventing content.

Stitching modes:

- `codex` — structured Codex CLI run; the development default;
- `command` — any local or hosted LLM CLI that reads a prompt from stdin and
  prints the requested JSON;
- `deterministic` — removes exact repeated suffix/prefix tokens without an LLM;
- `none` — keeps raw overlap text and therefore may contain duplicates.

Raw ASR responses are always retained in the manifest. See
[Architecture](docs/ARCHITECTURE.md) and [Known limitations](docs/LIMITATIONS.md).

## Common recipes

### OpenAI-compatible audio endpoint

```bash
export ASR_URL=https://your-asr-host.example/v1/audio/transcriptions
export ASR_API_KEY=replace-me

video-subtitle-pipeline video.mp4 \
  --asr-provider openai-compatible \
  --asr-mode whole \
  --asr-model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr-language en-US \
  --allow-audio-upload \
  --translate-provider codex \
  --target-language Korean \
  --include-source \
  --burn
```

Remote HTTP ASR refuses to run without `--allow-audio-upload`. Loopback URLs do
not require that flag. Only a temporary mono 16 kHz WAV is sent, never the video
container or frames. Whole mode requires a verbose response with `words`,
`start`, and `end`; use `--asr-mode segmented` for text-only endpoints.

Use `--asr-language en-US` when the spoken language is known. `auto` is the
portable default, but explicit Nemotron prompt conditioning can materially
improve recognition.

### Nemotron throughput and streaming policy

For the bundled Transformers server, whole-file requests can select how the
model executes without giving up word timestamps or optional confidence:

```bash
video-subtitle-pipeline video.mp4 \
  --asr-mode whole \
  --asr-runtime auto \
  --asr-memory-limit-gb 18 \
  --asr-max-offline-seconds 900 \
  --asr-confidence
```

- `auto` uses offline full-spectrogram encoding when the provider estimates it
  fits both current free VRAM and the supplied budget; otherwise it streams.
- `throughput` removes the provider's automatic duration cap, while still
  honoring explicit memory and duration limits.
- `streaming` always uses fixed-size cache-aware chunks and bounded encoder
  memory.
- `provider-default` sends no nonstandard runtime field and is the portable
  default for other OpenAI-compatible ASR services. The ignored local config
  for this deployment selects `auto` explicitly.

If offline allocation still runs out of memory, the bundled server clears the
failed allocation and retries with streaming. Its verbose response records the
requested/actual runtime, selection reason, fallback, elapsed time, and peak
allocated/reserved GPU memory; the pipeline preserves this metadata in the
manifest. By default it releases unused offline cache after each request so the
peak does not become steady GPU occupancy.
The memory calculation is an estimate calibrated through environment values,
not a CUDA allocation guarantee.

The HTTP endpoint currently returns one final JSON response. Streaming mode
reduces memory and uses the model's low-latency execution path, but progressive
partial captions over SSE/WebSocket remain future work. Likewise, true batch
size means multiple independent recordings or live streams; it is distinct
from the duration of one offline input.

## Confidence-assisted frame review

The bundled Transformers server can attach an RNNT maximum-softmax score to
each emitted token and aggregate a word's score with `min`. These values are
useful ranking signals, not calibrated probabilities that a word is correct.

```bash
video-subtitle-pipeline video.mp4 \
  --asr-mode whole \
  --asr-confidence \
  --visual-review-provider codex \
  --visual-review-model gpt-5.6-luna \
  --visual-review-confidence-threshold 0.6 \
  --allow-frame-upload
```

For each selected anchor, the pipeline samples a frame at its lowest-confidence
word and sends bounded transcript/neighbor context plus the frame through
`codex exec --image`. Proposal-only is the default. Raw ASR remains immutable,
and the manifest records the proposal, rationale, evidence class, frame time,
and SHA-256 hash.

`--visual-review-apply` is an explicit higher-risk opt-in. Generic
`visual_context` proposals are never applied, but vision models can still
misread or invent `visible_text`; human validation remains necessary.

Current OpenAI models, including GPT-5.6 Luna, support image input:
https://developers.openai.com/api/docs/models

### Segmented compatibility mode

```bash
video-subtitle-pipeline video.mp4 \
  --asr-mode segmented \
  --audio-overlap 1 \
  --asr-workers 4 \
  --stitch-provider codex \
  --use-source-text
```

### Local NeMo-Speech.cpp

```bash
video-subtitle-pipeline video.mp4 \
  --asr-provider command \
  --asr-mode segmented \
  --asr-model /path/to/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --asr-command 'nemo-speech transcribe {audio} --model {model} --format json --word-times' \
  --audio-overlap 1 \
  --stitch-provider codex \
  --use-source-text
```

`{audio}` and `{model}` are substituted as individual command arguments;
commands are not executed through a shell.

### Local Transformers adapter

Install PyTorch and a current Transformers release appropriate for your
hardware, then:

```bash
video-subtitle-pipeline video.mp4 \
  --asr-provider command \
  --asr-mode segmented \
  --asr-command 'python3 providers/nemotron_transformers.py {audio} --model {model}' \
  --asr-model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --use-source-text
```

The adapter imports Transformers only when invoked; it is not a core dependency.
For a timestamp/confidence-capable HTTP deployment, use
`providers/nemotron_transformers_server.py`; it exposes offline, automatic, and
cache-aware streaming execution with OpenAI-style verbose word timestamps plus
optional decoder confidence.

### Local LLM through Ollama

```bash
video-subtitle-pipeline video.mp4 \
  --asr-provider command \
  --asr-command 'nemo-speech transcribe {audio} --model {model} --format json --word-times' \
  --asr-model /path/to/model.gguf \
  --stitch-provider command \
  --translate-provider command \
  --llm-command 'python3 providers/ollama_json.py --model {model}' \
  --stitch-model qwen3:8b \
  --translation-model qwen3:8b \
  --target-language Korean \
  --include-source
```

The generic `providers/openai_chat_json.py` adapter works with local LM Studio,
llama.cpp, or hosted OpenAI-compatible chat endpoints via `LLM_BASE_URL`,
`LLM_API_KEY`, and `--model`.

## Existing anchors and corrections

Use an SRT or a previous manifest as the source anchors:

```bash
video-subtitle-pipeline video.mp4 \
  --anchors reviewed.segments.json \
  --allow-empty-text \
  --translate-provider codex \
  --include-source \
  --burn
```

Existing source text is not sent to ASR. Existing translations are not
regenerated. One-off translations can be supplied with repeatable
`--translated-segment 'ID=TEXT'` arguments.

## Provider and platform setup

See [Provider setup](docs/PROVIDERS.md) for:

- Nemotron 3.5 on Linux, Windows, macOS, NVIDIA GPU, Apple Silicon, and CPU;
- NeMo-Speech.cpp, Transformers, and NVIDIA NIM tradeoffs;
- Codex CLI, Ollama, llama.cpp, and LM Studio translation paths;
- hosted services with free experimentation allocations.

## Repository layout

```text
src/video_subtitle_pipeline/   Core pipeline and CLI
providers/                     Optional command-provider adapters
tests/                         Standard-library unit tests
docs/                          Architecture, providers, limitations, roadmap
config.example.json            Safe committed configuration example
config.local.json              Private ignored machine configuration
AGENTS.md                      Contributor and coding-agent guidance
```

Run validation with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src providers tests
```

## Project status

This is a working personal-project pipeline, not a polished media production
suite. Whole-file native word timestamps avoid artificial subtitle-boundary
cuts, while segmented overlap remains a compatibility fallback. Neither mode
replaces human review: technical names, acronyms, quiet speech, multiple
speakers, and cross-talk remain likely correction points.

Soft/native English and translated subtitle tracks are recorded as future work
in [Future work](docs/FUTURE_WORK.md). Hard subtitles remain the most universally
visible delivery format.

No license has been selected yet. Choose one before publishing the repository.
