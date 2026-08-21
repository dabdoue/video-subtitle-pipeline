# Architecture

## Pipeline stages

```text
video
  -> probe streams/duration
  -> nominal anchors
  -> one continuous audio track
  -> stateful ASR + native word timestamps
  -> timestamp assignment into nominal anchors
  -> optional low-confidence frame review proposals
  -> optional translation
  -> timed cues
  -> SRT + manifest
  -> optional hard-sub MP4
```

The source video is never split or concatenated. Whole mode extracts one
temporary mono 16 kHz WAV. Segmented compatibility mode extracts overlapping
WAV windows. Hard-sub rendering re-encodes the video stream and copies the
selected original audio stream.

## Whole-file timestamp assignment

Whole mode is the default. A timestamp-capable provider receives the complete
audio track and returns ordered words with start/end seconds. Each word is
assigned to the nominal anchor containing its temporal midpoint. The raw words,
their times, and the complete audio bounds remain in the manifest.

The bundled Nemotron Transformers server separates timestamp generation from
execution policy. Offline mode passes one full feature tensor and encodes the
spectrogram up front. Streaming mode passes a generator of fixed-size features
and preserves encoder attention/convolution caches across chunks. Both paths
return RNNT token durations, so both produce timestamps and optional confidence.

`auto` estimates offline incremental memory from input duration, current free
VRAM, a reserved margin, and optional user limits. It selects offline when the
estimate fits and streaming otherwise. A CUDA out-of-memory failure during an
offline attempt triggers one streaming retry. The runtime decision and actual
path remain in the ASR response and final manifest.

Streaming keeps bounded encoder activations. Its padded final inference chunk
is clipped to the true media duration before transcript text is reconstructed.

## Decoder confidence

The bundled server captures the maximum softmax value at each greedy RNNT
generation step. Blank/special emissions are removed, remaining scores align
with timestamped emitted tokens, and word confidence is the minimum of its
token scores. The response labels this `rnnt_max_softmax` with `min`
aggregation and records that it is not a calibrated probability.

## Frame-assisted review

Segments with a word below the configured threshold are grouped into bounded
batches. One PNG per segment is sampled at the lowest-confidence word midpoint.
A Codex vision call receives frames, source/neighbor text, timing, and scores
through a strict output schema.

Each proposal records frame time/hash, model, threshold, original/proposed
text, rationale, evidence class, and apply decision. `raw_asr_text` and
`asr_words` never change. Proposal-only is the default; generic visual context
cannot auto-apply even when applying is explicitly enabled.

## Segmented nominal anchors versus ASR windows

Subtitle anchors remain contiguous and non-overlapping. With a nominal interval
`[start, end]` and overlap `o`, ASR receives:

```text
[max(0, start - o), min(video_duration, end + o)]
```

For five-second anchors and one-second overlap, adjacent ASR windows share two
seconds of audio. This is intentional: each boundary is audible with a second of
context on both sides.

## Stitching

The LLM stitcher receives raw ASR text plus nominal/extraction timestamps. It
also receives one neighboring record outside each output batch when available.
Only target IDs appear in the required output schema.

The stitcher is instructed to:

- assign overlap speech exactly once;
- reconstruct partial boundary words only from neighboring evidence;
- retain technical wording and uncertainty;
- avoid translation, grammar rewriting, and invented content;
- emit empty text for genuine silence.

Schema validation enforces the exact ID set. Invalid JSON, duplicate IDs, and
missing IDs trigger up to three attempts and then stop the pipeline.

The deterministic stitcher is deliberately conservative. It removes only the
longest exact normalized suffix/prefix token match. It cannot decide where a
non-identical paraphrased overlap belongs and may shift a phrase by as much as
the overlap duration.

## Provenance model

Each manifest segment has:

```json
{
  "id": "0002",
  "start": 5.0,
  "end": 10.0,
  "raw_asr_text": "...",
  "asr_audio_start": 4.0,
  "asr_audio_end": 11.0,
  "asr_words": [{"word": "robot", "start": 5.1, "end": 5.4, "confidence": 0.42}],
  "text": "stitched source text",
  "visual_review": null,
  "translation_parts": ["translated text"]
}
```

Reviewers can correct `text` or `translation_parts` without destroying the raw
evidence. Reusing that manifest as `--anchors` skips already supplied work.

## Concurrency and failure handling

Whole mode makes one ASR request. The current server serializes inference on a
single model instance because its generation state and confidence hook are not
safe to mutate concurrently. Segmented ASR windows use a bounded thread
pool; results are assigned by segment object, not completion order. Failed
parallel requests are retried once sequentially; a second failure stops the
run.

LLM batches are sequential by design so validation failures and provider rate
limits remain understandable. Stitch batches include neighbor context across
batch boundaries.

## Configuration precedence

```text
explicit CLI flag > JSON config > environment/default
```

An explicit `--config` wins over automatic repository-local discovery.
`VIDEO_SUBTITLE_CONFIG` is used when no flag is supplied. Finally, an existing
`config.local.json` in the current directory is loaded.
