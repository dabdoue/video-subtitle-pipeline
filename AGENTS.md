# AGENTS.md

Guidance for humans and coding agents working in this repository.

## Scope and invariants

- This repository is a standalone video transcription/translation tool. Do not
  add dependencies on the original Modelythic router checkout.
- Whole-file ASR with native word timestamps, memory-aware offline/streaming
  selection, five-second nominal anchors, Korean translation, bilingual
  subtitles, and optional hard-sub rendering are the reference workflow.
  One-second overlap, four workers, and validated stitching remain the
  segmented compatibility workflow.
- Nominal subtitle times must never overlap. Audio extraction windows may
  overlap and must be clamped to `[0, video_duration]`.
- Preserve raw ASR, word times/confidence, reviewed source text, review
  proposals, and translated text as separate facts. Never overwrite raw
  evidence with a corrected value.
- LLM stitching and translation must validate exact segment IDs, reject partial
  output, and retry boundedly. Never silently accept missing or duplicate IDs.
- A failed ASR/LLM/render step must produce a nonzero exit rather than a
  plausible-looking partial deliverable.

## Privacy and configuration

- Never commit API keys, private hostnames, personal absolute paths, or the
  contents of `config.local.json`/`.env`.
- `config.example.json` must remain runnable as documentation but use only
  placeholder hosts and generic environment names.
- Remote audio upload requires explicit approval through
  `--allow-audio-upload`; local command and loopback providers do not.
- Remote frame review requires separate `--allow-frame-upload`. Proposal-only
  is the safe default; visual-context proposals must never auto-apply.
- Manifests may record provider/model identity but must not record credentials
  or the private endpoint URL.

## Provider contracts

- Whole-mode OpenAI-compatible ASR returns `text` plus timestamped `words`;
  confidence and runtime metadata are optional. Segmented mode accepts a
  text-only JSON response. Nonstandard runtime form fields must remain opt-in
  for compatibility with generic OpenAI-style providers.
- Command ASR templates support `{audio}` and `{model}` and must print either
  plain transcript text or JSON containing `text`/`transcript`.
- Command LLM providers receive the complete prompt on stdin, support `{model}`
  in their argument template, and print the requested JSON object on stdout.
- Keep core dependencies at Python standard library plus external CLIs. Optional
  heavy runtimes belong in `providers/` adapters.

## Codex CLI setup

- Treat OpenAI's standalone Codex CLI as the supported runtime for the `codex`
  stitching, translation, and visual-review providers. Install it with the
  current official installer and verify both `codex --version` and
  `codex login status` before an expensive pipeline run.
- Authentication is a user-level prerequisite. A human must complete
  `codex login` when no valid cached session exists. Never automate browser
  consent, solicit credentials in chat, or read, copy, print, or commit
  `~/.codex/auth.json`; it contains sensitive access tokens.
- `run-local.sh` must prefer the standalone `~/.local/bin/codex`, then a normal
  `PATH` installation. Discovery of a Codex binary bundled with the OpenAI VS
  Code extension is a fallback for continuity, not the documented installation
  path.
- Codex CLI and the IDE extension can share cached login details, but do not
  assume that installing one authenticates the other. Check status explicitly
  and fail nonzero when the requested provider is unavailable.

## Editing workflow

1. Inspect `git status --short` before editing. Preserve unrelated user work.
2. Update or add tests for behavior changes, especially overlap boundaries,
   config precedence, provider contracts, and manifest provenance.
3. Run:

   ```bash
   PYTHONPATH=src python3 -m unittest discover -s tests -v
   python3 -m compileall -q src providers tests
   ```

4. For changes touching FFmpeg commands, run a short generated-media smoke test
   and inspect the resulting SRT/manifest. Do not claim visual validation from
   metadata checks alone.
5. Do not automatically commit. Suggest a focused commit message after the user
   tests the workflow.

## Documentation expectations

- Keep README commands copyable and platform assumptions explicit.
- Link to primary vendor/model documentation. Free tiers and model support can
  change; state that instead of promising permanent availability.
- Add known quality or hardware limitations to `docs/LIMITATIONS.md` rather
  than hiding them in implementation comments.
- Add deferred product features to `docs/FUTURE_WORK.md` with acceptance notes.
