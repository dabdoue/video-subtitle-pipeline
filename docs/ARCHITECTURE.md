# Architecture

## Pipeline stages

```text
video
  -> probe streams/duration
  -> nominal anchors
  -> overlapping audio windows
  -> raw ASR transcripts
  -> boundary stitching
  -> source transcript per nominal anchor
  -> optional translation
  -> timed cues
  -> SRT + manifest
  -> optional hard-sub MP4
```

The source video is never split or concatenated. Only temporary mono 16 kHz WAV
files are extracted. Hard-sub rendering re-encodes the video stream and copies
the selected original audio stream.

## Nominal anchors versus ASR windows

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
  "text": "stitched source text",
  "translation_parts": ["translated text"]
}
```

Reviewers can correct `text` or `translation_parts` without destroying the raw
evidence. Reusing that manifest as `--anchors` skips already supplied work.

## Concurrency and failure handling

ASR windows use a bounded thread pool. Results are assigned by segment object,
not completion order. Failed parallel requests are retried once sequentially;
a second failure stops the run.

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

