# Provider setup

Provider availability, free allocations, and hardware support change. Check the
linked primary documentation before relying on a path for production.

## ASR: NVIDIA Nemotron 3.5 locally

Official model:

- Model card and files:
  https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- NeMo-Speech.cpp:
  https://github.com/NVIDIA/NeMo-Speech.cpp
- NeMo-Speech.cpp installation:
  https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/install.md
- CLI and JSON output:
  https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/cli.md

The model is approximately 0.6B parameters. NVIDIA publishes full checkpoints
and a roughly 742 MB Q8 GGUF artifact. Review the model license on its model card
before redistribution.

### NeMo-Speech.cpp: Linux and macOS

Clone the official repository, inspect its installer, and run:

```bash
scripts/install.sh --source
export PATH="$HOME/.local/bin:$PATH"
nemo-speech --version
```

The installer selects CUDA when `nvidia-smi` is available, Metal on Apple
Silicon, and CPU otherwise. A source build requires Git, CMake 3.26+, Ninja, a
C++17 compiler, and the chosen GPU toolkit.

Download the published GGUF from the model card:

```bash
hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
  nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --local-dir models
```

Verify a WAV directly:

```bash
nemo-speech transcribe audio.wav \
  --model models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --json
```

Backends can be selected with `--device cuda:0`, `--device metal`,
`--device vulkan:0`, or `--device cpu`. CPU is the most portable and generally
the slowest; Q8 is the practical compact default.

### NeMo-Speech.cpp: Windows

From an official source checkout in PowerShell:

```powershell
.\scripts\install.ps1 -Source
nemo-speech --version
```

For NVIDIA CUDA:

```powershell
.\scripts\install.ps1 -Source -Backend cuda
```

The documented source prerequisites are Git, CMake 3.26+, Ninja, Visual Studio
2022 Build Tools with C++, and the selected backend toolkit. Vulkan and CPU are
also installer options.

### Local HTTP server

NeMo-Speech.cpp can expose an OpenAI-compatible transcription subset:

```bash
nemo-speech serve \
  --asr-model models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
```

The documented default is `http://127.0.0.1:8080`; configure the pipeline with:

```json
{
  "asr_provider": "openai-compatible",
  "asr_url": "http://127.0.0.1:8080/v1/audio/transcriptions",
  "asr_model": "default"
}
```

Client documentation:
https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/clients.md

### Transformers

The official model card documents Transformers support. Install a PyTorch build
appropriate for CUDA, Apple MPS, or CPU, then install the model-card minimum
Transformers version. This repository supplies `providers/nemotron_transformers.py`
as a command adapter.

This path is convenient for Python environments but downloads a larger full
checkpoint and has a heavier dependency stack than the GGUF runtime.

### NVIDIA NeMo Speech framework

The model card also supports:

```python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-3.5-asr-streaming-0.6b"
)
print(model.transcribe(["audio.wav"]))
```

Use the current NeMo Speech installation guide rather than an old pinned command:
https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/starthere/install.rst

### NVIDIA Speech NIM

NIM is a supported container/server path, but it is not the simplest free local
path. NVIDIA's current prerequisites require an NVIDIA AI Enterprise license,
x86_64 Linux/container support, and model-specific GPU requirements. The ASR
support matrix currently lists at least 16 GB VRAM and compute capability 8.0+
for the service family; Nemotron ASR Streaming is not listed as WSL2-compatible.

- Prerequisites: https://docs.nvidia.com/nim/speech/latest/get-started/prerequisites.html
- ASR support matrix: https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html

For personal/local use, NeMo-Speech.cpp or Transformers is usually the more
accessible starting point.

## Other local ASR options

The command provider is model-agnostic. A wrapper only needs to accept a WAV
path and print plain text or `{"text":"..."}`. That makes it suitable for:

- whisper.cpp: https://github.com/ggml-org/whisper.cpp
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- a custom NeMo script;
- any local OpenAI-compatible audio server.

Different ASR models have different language, punctuation, timestamp, and
diarization behavior. Do not assume settings tuned for Nemotron transfer exactly.

## Hosted ASR for experimentation

Groq currently documents an OpenAI-compatible speech-to-text endpoint, Whisper
Large V3/V3 Turbo models, and a Free tier with a 25 MB per-file limit. It can be
used directly with the HTTP provider:

```bash
export ASR_URL=https://api.groq.com/openai/v1/audio/transcriptions
export ASR_API_KEY=replace-with-groq-key

video-subtitle-pipeline video.mp4 \
  --asr-provider openai-compatible \
  --asr-model whisper-large-v3-turbo \
  --allow-audio-upload \
  --use-source-text
```

The pipeline's short mono WAV windows stay well below the documented per-file
limit. Free-tier quotas and supported models may change.

- Speech-to-text documentation: https://console.groq.com/docs/speech-to-text
- Billing/free-tier FAQ: https://console.groq.com/docs/billing-faqs

This is a Whisper alternative, not hosted Nemotron 3.5. Use local Nemotron or a
Nemotron-capable endpoint when model consistency matters.

## Stitching and translation locally

### Codex CLI

The `codex` provider uses ephemeral, read-only `codex exec` calls and validates
output against a generated JSON schema. The model must be available to the
installed Codex configuration. Model names are intentionally configurable.

### Ollama

Ollama supports Windows, macOS, and Linux and its generation API supports JSON
or JSON-schema output:

- API documentation: https://docs.ollama.com/api/generate

Use the included adapter:

```text
--stitch-provider command
--translate-provider command
--llm-command "python3 providers/ollama_json.py --model {model}"
```

Choose a multilingual instruction model that reliably follows JSON constraints.
Small models are faster but more likely to lose IDs or rewrite technical text.

### llama.cpp

llama.cpp runs quantized LLMs across CPU and GPU and provides an OpenAI-compatible
server with schema-constrained JSON:

- Project and quick start: https://github.com/ggml-org/llama.cpp
- Server documentation:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Run `providers/openai_chat_json.py` against its local `/v1` endpoint.

### LM Studio

LM Studio can run a local server through its GUI or `lms server start`, with
OpenAI-compatible endpoints and structured output:

- Local server: https://lmstudio.ai/docs/developer/core/server
- Structured output:
  https://lmstudio.ai/docs/developer/openai-compat/structured-output

Point `LLM_BASE_URL` at the local `/v1` endpoint and use the generic adapter.

## Hosted/free experimentation

The pipeline does not hard-code a hosted translation vendor. The command adapter
can wrap any API that returns the requested JSON. Current starting points include:

- Hugging Face Inference Providers gives free users a small monthly credit,
  explicitly subject to change:
  https://huggingface.co/docs/inference-providers/en/pricing
- Cloudflare Workers AI has a daily free allocation on its Free and Paid plans:
  https://developers.cloudflare.com/workers-ai/platform/pricing/
- OpenRouter exposes `openrouter/free` and `:free` model variants with lower
  rate limits and variable availability:
  https://openrouter.ai/docs/guides/routing/routers/free-router

These are useful for experimentation, not guaranteed throughput. Confirm that a
chosen model supports the target language and reliable structured JSON. Review
each provider's privacy and data-retention terms before sending transcripts.

For hosted ASR, use an OpenAI-compatible audio endpoint or add a small command
adapter for the provider's SDK. Free model hosting may be too limited for long
videos; local inference avoids per-request uploads and changing quotas.
