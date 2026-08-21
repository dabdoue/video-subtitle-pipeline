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

## Word-timestamp or stateful ASR

- Accept verbose ASR responses with word timestamps and use those timestamps to
  assign overlap text deterministically.
- Add provider capabilities for frame-, token-, and word-level ASR confidence.
  NeMo RNNT decoding can be configured to preserve these scores, but the
  default Nemotron response is text-only and not every serving adapter exposes
  the richer hypothesis data.
- Preserve word confidence and its calculation metadata in the manifest,
  including the method (`max_prob` or normalized entropy), word aggregation,
  model/runtime version, and whether the value came directly from the decoder.
- Use confidence to flag likely review points and prioritize context-assisted
  correction. Do not treat a decoder score as a calibrated probability that a
  word is correct; establish useful thresholds against reviewed, in-domain
  recordings first.
- Support one stateful streaming session across the complete audio track for
  models such as Nemotron 3.5, while retaining nominal subtitle anchors.
- Compare independent-window overlap against cache-aware streaming quality.

Useful references:

- NeMo ASR timestamps:
  https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html#obtain-timestamps
- NeMo transcription script (`confidence=true` emits token and word confidence):
  https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/transcribe_speech.py
- NeMo confidence configuration and supported score methods:
  https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/utils/asr_confidence_utils.py

## Frame-assisted transcript review

Status: documented, not implemented.

- Extract a representative video frame for each selected segment, initially at
  its nominal midpoint, with a later scene-aware option to avoid transitions or
  blank frames.
- Pass the frame, raw overlapping ASR, neighboring source text, timestamps, and
  glossary to a vision-capable Luna/Codex review call. Visible application tabs,
  labels, diagrams, and demonstrated objects can then support corrections such
  as a domain term that sounds similar to a common word.
- Make this provider-capability-driven: retain a text-only review fallback when
  the selected model or CLI path cannot accept images.
- Ask the reviewer for structured correction proposals containing the segment
  ID, original text, proposed text, rationale, and evidence type (`audio`,
  `neighbor`, `visible_text`, `visual_context`, or `glossary`).
- Never replace `raw_asr_text`. Apply a visual correction only to the reviewed
  source-text field, and retain the proposal and provenance in the manifest.
- Treat visible text as supporting context, not proof of speech. A frame may
  show a nearby control or stale screen that the speaker did not actually name.
- Prefer sending frames only for low-confidence words, glossary conflicts, or
  manually selected segments so review cost and image disclosure stay bounded.

Acceptance notes:

- The sampled frame timestamp and a reproducible frame hash are recorded.
- Corrections are traceable to raw ASR and can be accepted, rejected, or
  reverted without retranscription.
- A validation fixture covers a visually disambiguated domain term and another
  case where misleading on-screen text must not change the transcript.
- Private videos are not sent to a remote vision provider without the same
  explicit upload permission used for remote audio.

## Human review workflow

- Generate a lightweight HTML reviewer with waveform, source video, raw ASR,
  stitched text, translation, and per-segment approval.
- Add glossary locks for product names, labware, chemical abbreviations, names,
  and units.
- Allow correction export back into the manifest and one-click rerendering.

## Operational improvements

- Incremental checkpoints and resume after interruption.
- Directory/ZIP batch command with bounded disk usage.
- Provider health checks and capability discovery.
- Speaker diarization and accessible sound-event captions.
- Quality metrics for duplicate overlap, missing translations, line density,
  reading speed, and suspicious script/language changes.
- Preserve multiple audio tracks, chapters, HDR/color metadata, and selected
  camera metadata when rendering.
