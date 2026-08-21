# Future work

## Native soft subtitle tracks

Status: documented, not implemented.

Generate separate language sidecars from the manifest:

```text
video.en.srt
video.ko.srt
video.bilingual.srt
video.en.vtt
video.ko.vtt
```

This would let YouTube and similar platforms expose native selectable captions
instead of requiring burned text. SRT is the practical broad-compatibility
format; WebVTT is the W3C web-native timed-text format.

Useful references:

- YouTube supported caption files:
  https://support.google.com/youtube/answer/2734698
- YouTube language-track upload workflow:
  https://support.google.com/youtube/answer/2734796
- W3C WebVTT specification:
  https://www.w3.org/TR/webvtt1/

Acceptance notes:

- Produce English, translated, and optional bilingual tracks without ASR/LLM
  reruns.
- Validate UTF-8, monotonic cue times, language tags, and cue coverage.
- Keep burned output as an independent option.
- Investigate optional embedded MP4 `mov_text`, but do not treat it as the
  portable website delivery format.

## ASR confidence and alternate streaming providers

- Extend confidence support beyond the bundled Nemotron Transformers server.
- Add optional beam/N-best sequence decoding and auditable word alternatives.
  Greedy top-token scores are not a word candidate list: in one validation, the
  `wa` branch needed for `labware` ranked 40th at the `labor` divergence point,
  so a small top-k list would not have recovered it.
- Compare custom greedy maximum-softmax confidence with NeMo's configured
  maximum-probability and entropy confidence methods on reviewed data.
- Extend native whole-file timestamps and stateful streaming beyond the bundled
  Nemotron Transformers server to command and hosted providers.
- Benchmark independent-window overlap against cache-aware streaming quality.

Useful references:

- NeMo ASR timestamps:
  https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html#obtain-timestamps
- NeMo transcription script (`confidence=true` emits token and word confidence):
  https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/transcribe_speech.py
- NeMo confidence configuration and supported score methods:
  https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/utils/asr_confidence_utils.py

## Frame-assisted transcript review

Status: initial confidence-selected Codex image review implemented.

- Add scene-aware multi-frame sampling to avoid transitions, motion blur, and a
  misleading single instant.
- Add independent OCR verification before any visible-text proposal can become
  eligible for automatic application.
- Make this provider-capability-driven: retain a text-only review fallback when
  the selected model or CLI path cannot accept images.
- Add glossary locks and manual segment selection alongside confidence-based
  targeting.

Acceptance notes:

- Build a reviewer UI to accept/reject proposals and rerender without
  retranscription.
- Validate multi-frame/OCR gating against both a true visible correction and a
  misleading on-screen term.

## Human review workflow

- Generate a lightweight HTML reviewer with waveform, source video, raw ASR,
  stitched text, translation, and per-segment approval.
- Add glossary locks for product names, labware, chemical abbreviations, names,
  and units.
- Allow correction export back into the manifest and one-click rerendering.

## Operational improvements

- Incremental checkpoints and resume after interruption.
- Directory/ZIP batch command with bounded disk usage.
- Add a multi-request scheduler that batches independent recordings/streams on
  the GPU; keep this distinct from single-recording offline input duration.
- Add SSE or WebSocket partial transcripts for clients that need visible
  real-time feedback from the streaming runtime.
- Provider health checks and capability discovery.
- Speaker diarization and accessible sound-event captions.
- Quality metrics for duplicate overlap, missing translations, line density,
  reading speed, and suspicious script/language changes.
- Preserve multiple audio tracks, chapters, HDR/color metadata, and selected
  camera metadata when rendering.
