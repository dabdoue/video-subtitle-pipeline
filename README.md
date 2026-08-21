# Video Subtitle Pipeline

A repeatable, auditable workflow for turning a video into timed source
transcripts, translated subtitles, and an optional hard-subtitled MP4.

The default workflow uses five-second subtitle anchors. Each ASR request may
include one second of extra audio before and after its nominal anchor, then a
stitching stage reconciles the overlapping transcripts so boundary words are
not cut or repeated.

## What it produces

For `training.mp4` and Korean output (`--language-code ko`):

```text
training.ko.srt
training.ko.segments.json
training.ko.hardsub.mp4   # only with --burn
```

The JSON manifest is the source of truth. It retains:

- nominal subtitle ranges;
- the wider audio range sent to ASR;
- raw overlapping ASR text;
- stitched source text;
- translation parts and rendered cue timing;
- provider/model names without credentials or the endpoint URL.

That separation makes errors reviewable and lets captions be rerendered from a
corrected manifest without guessing what happened at a boundary.

## Install

Requirements:

- Python 3.11 or newer;
- `ffmpeg` and `ffprobe`;
- `curl` for an OpenAI-compatible ASR endpoint;
- optionally `codex` for Codex CLI stitching/translation;
- a target-language font, such as Noto CJK, for hard subtitles.

Install the CLI in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
video-subtitle-pipeline --help
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

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

## The overlap workflow

With five-second anchors and `--audio-overlap 1`, nominal segment `5–10s` is
transcribed from audio `4–11s`. Its neighbors are `0–6s` and `9–16s`.

```text
nominal anchors:  [ 0-----5 ][ 5----10 ][10----15 ]
ASR windows:      [ 0--------6 ]
                         [ 4--------11 ]
                                  [ 9--------16 ]
```

The stitcher sees raw text from neighboring windows and returns one clean
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
  --asr-model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --allow-audio-upload \
  --audio-overlap 1 \
  --stitch-provider codex \
  --translate-provider codex \
  --target-language Korean \
  --include-source \
  --burn
```

Remote HTTP ASR refuses to run without `--allow-audio-upload`. Loopback URLs do
not require that flag. Only temporary mono 16 kHz WAV windows are sent, never
the video itself.

### Local NeMo-Speech.cpp

```bash
video-subtitle-pipeline video.mp4 \
  --asr-provider command \
  --asr-model /path/to/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --asr-command 'nemo-speech transcribe {audio} --model {model} --json' \
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
  --asr-command 'python3 providers/nemotron_transformers.py {audio} --model {model}' \
  --asr-model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --use-source-text
```

The adapter imports Transformers only when invoked; it is not a core dependency.

### Local LLM through Ollama

```bash
video-subtitle-pipeline video.mp4 \
  --asr-provider command \
  --asr-command 'nemo-speech transcribe {audio} --model {model} --json' \
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
suite. Five-second anchors plus one-second overlap improve boundary context but
do not replace word-level timestamps or human review. Technical names, acronyms,
quiet speech, multiple speakers, and cross-talk remain likely correction points.

Soft/native English and translated subtitle tracks are recorded as future work
in [Future work](docs/FUTURE_WORK.md). Hard subtitles remain the most universally
visible delivery format.

No license has been selected yet. Choose one before publishing the repository.

