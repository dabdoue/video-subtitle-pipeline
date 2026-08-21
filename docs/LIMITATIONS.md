# Known limitations and operational notes

## Transcription quality

- Overlap improves boundary context, but the ASR response currently lacks
  word-level timestamps. The stitcher infers ownership from duplicate text and
  neighboring context; it does not listen to the audio itself.
- ASR systems may emit different spellings for the same overlapped phrase. An
  LLM can reconcile many cases, but neither the LLM nor deterministic fallback
  can guarantee the true wording.
- Technical names, acronyms, units, quiet speech, room noise, cross-talk, and
  speakers far from the microphone require human review.
- A small anchor improves synchronization but supplies less semantic context.
  One-second overlap is a practical default, not a universal optimum.
- Streaming Nemotron models are designed to carry encoder state across chunks.
  Independent HTTP/CLI requests do not preserve that state; overlap is a
  workaround, not equivalent streaming inference.

## Stitching and translation

- LLM output can hallucinate or normalize terminology despite strict prompts.
  The manifest retains raw ASR output specifically so this remains auditable.
- Deterministic stitching only removes exact token overlap and can leave
  paraphrased duplicates.
- Translation operates in bounded batches. It does not currently maintain a
  project glossary or whole-video terminology memory.
- Free hosted models can change, rate-limit, disappear, or return weaker JSON.
  Structured-output reliability matters as much as translation quality.

## Timing

- Each stitched source segment is assigned to its original nominal anchor. The
  pipeline does not generate word- or phoneme-level timing.
- Multi-part translations are split evenly inside the anchor. Pause-aware cue
  splitting from the original prototype is not included in this standalone
  version because it was not reliable enough to keep as a default.
- Silence can produce empty segments; subtitle cue counts therefore need not
  equal anchor counts.
- Generated container tails shorter than `--minimum-anchor-seconds` are merged
  into the preceding anchor so rounding artifacts do not become standalone cues.

## Media behavior

- Hard subtitles require video re-encoding. The selected audio stream is copied.
- Only one audio stream is included in the hard-sub output. Additional audio,
  data, GPS, HDR, chapters, and camera metadata are not preserved.
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

- Remote ASR receives extracted audio windows. The video itself is not uploaded.
- Translation/stitch providers receive transcript text, timing, and model
  prompts. They do not receive video or audio unless a custom provider does so.
- Local config and `.env` are ignored, but users must still inspect `git status`
  before publishing.
