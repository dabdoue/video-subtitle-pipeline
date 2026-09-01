# Known limitations and operational notes

## Transcription quality

- Whole mode depends on a provider that exposes genuine word timestamps. A
  text-only or superficially OpenAI-compatible endpoint must use segmented mode.
- Timestamp accuracy is limited to the ASR encoder frame rate and word
  aggregation. It is alignment evidence, not a human-verified boundary.
- Decoder confidence is not calibrated. Thresholds must be tuned against
  reviewed recordings for the particular model, language, microphone, and
  domain.
- ASR systems may emit different spellings for the same overlapped phrase. An
  LLM can reconcile many cases, but neither the LLM nor deterministic fallback
  can guarantee the true wording.
- Technical names, acronyms, units, quiet speech, room noise, cross-talk, and
  speakers far from the microphone require human review.
- A small anchor improves synchronization but supplies less semantic context.
  One-second overlap is a practical default, not a universal optimum.
- The bundled Nemotron server preserves encoder state across internal chunks.
  Independent segmented HTTP/CLI requests do not; overlap remains a workaround
  for those providers.
- Offline memory selection is estimated from a configurable fixed cost and
  duration slope. Other audio content, library versions, attention backends,
  concurrent GPU work, and allocator fragmentation can change the true peak.
  The server falls back after a CUDA OOM, but operators should calibrate the
  estimate on their own hardware. The default estimate targets reserved VRAM,
  which can be materially higher than PyTorch's live allocated-tensor counter.
- The bundled endpoint currently serializes inference and returns completed
  JSON. It does not yet batch unrelated recordings or emit progressive partial
  captions over SSE/WebSocket.

## Stitching and translation

- LLM output can hallucinate or normalize terminology despite strict prompts.
  The manifest retains raw ASR output specifically so this remains auditable.
- Deterministic stitching only removes exact token overlap and can leave
  paraphrased duplicates.
- Translation operates in bounded batches. It does not currently maintain a
  project glossary or whole-video terminology memory.
- Free hosted models can change, rate-limit, disappear, or return weaker JSON.
  Structured-output reliability matters as much as translation quality.

## Visual review

- A single frame cannot establish what was spoken. It can show stale, nearby,
  or unrelated interface text.
- Vision models can misread or invent labels. In validation, one model call
  invented a useful-looking label while another grounded an incorrect
  transcript replacement in genuinely visible layout text.
- Proposal-only is therefore the default. Generic visual-context proposals are
  never auto-applied; visible-text proposals still require explicit opt-in and
  human validation.
- The frame may land during motion blur or a transition. Scene-aware sampling
  and independent OCR verification remain future work.

## Timing

- Whole mode assigns timestamped words to nominal anchors by word midpoint. It
  does not produce phoneme-level timing. Segmented mode retains anchor-level
  timing only.
- Multi-part translations are split evenly inside the anchor. Pause-aware cue
  splitting from the original prototype is not included in this standalone
  version because it was not reliable enough to keep as a default.
- Silence can produce empty segments; subtitle cue counts therefore need not
  equal anchor counts.
- Generated container tails shorter than `--minimum-anchor-seconds` are merged
  into the preceding anchor so rounding artifacts do not become standalone cues.

## Media behavior

- Hard subtitles require video re-encoding. The selected audio stream is copied.
- Only one audio stream is included in the hard-sub output. Additional audio
  and data streams are not mapped, but FFmpeg can still copy global container
  metadata, including iPhone location/GPS tags, into the rendered file. Inspect
  or strip metadata before sharing an output externally. HDR, chapters, and
  camera metadata are not guaranteed to survive re-encoding.
- The default H.264/yuv420p output is broadly compatible but not archival.
- Variable-frame-rate input may be normalized by FFmpeg during rendering.
- Font fallback depends on fonts installed on the machine. Missing glyphs may
  appear as boxes even when the SRT is correct.

## Reliability and resumability

- ASR and LLM results are written to the final manifest after processing. There
  is not yet an incremental checkpoint after every completed request.
- Re-running from a completed manifest is supported, but interruption during
  initial ASR can require retranscription.
- Disk use can be substantial during hard-sub encoding. The input, temporary
  WAVs, and partial output coexist.

## Security and privacy

- Remote whole-file ASR receives the extracted audio track; segmented ASR
  receives extracted windows. The video container itself is not uploaded.
- Translation/stitch providers receive transcript text, timing, and model
  prompts. They do not receive video or audio unless a custom provider does so.
- Visual review sends selected still frames only after `--allow-frame-upload`;
  it does not send the video container.
- Local config and `.env` are ignored, but users must still inspect `git status`
  before publishing.
