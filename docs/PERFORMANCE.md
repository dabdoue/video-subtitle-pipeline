# Performance notes

These are focused development measurements, not general model benchmarks.
Provider version, PyTorch/Transformers build, GPU clocks, audio content,
lookahead, concurrency, and request format can change the result.

## Original versus timestamp/confidence server

Measured on an otherwise idle NVIDIA RTX 3090 (24 GB) using the same fp16
Nemotron 3.5 ASR checkpoint. The original server used its text-only Transformers
pipeline and split inputs over 40 seconds into independent 30-second pieces.
The new server used one cache-aware stateful stream, lookahead 13, word
timestamps, and confidence unless noted.

The 10-second source clip was also repeated to create a synthetic 60-second
input. GPU memory was sampled through `nvidia-smi` every 100 ms, so very narrow
allocation spikes may be missed.

| Case | Original text-only | New timestamps + confidence | New timestamps, no confidence |
|---|---:|---:|---:|
| Cold 10 s, model load included | 8.367 s | 8.644 s | not repeated |
| Warm 10 s, mean of 3 | 0.321 s | 0.917 s | 0.899 s |
| Warm 60 s | 1.974 s | 4.585 s | 4.214 s |
| Steady GPU memory after 10 s | 1,588 MiB | 1,616 MiB | approximately 1,618 MiB |
| Sampled peak, warm 10 s | 1,662 MiB | 1,616 MiB | not separately sampled |
| Sampled peak, warm 60 s | 1,796 MiB | 1,618 MiB | approximately 1,618 MiB steady |

Interpretation:

- Cold time changed little because loading the 0.6B model dominates startup.
- Warm stateful timestamped inference was about 2.9x slower on 10 seconds and
  2.3x slower on 60 seconds than the original text-only/chunked implementation.
  It remained roughly 11-13x faster than real time with confidence enabled.
- Disabling confidence saved little on 10 seconds and about 0.37 seconds
  (roughly 8%) on the synthetic 60-second request. Stateful streaming and
  timestamp reconstruction account for most of the speed difference.
- New steady usage was about 28-30 MiB higher, but memory stayed essentially
  flat as duration grew. At 60 seconds its sampled peak was about 178 MiB lower
  than the original. This bounded behavior is the reason to prefer stateful
  streaming for full videos despite lower raw throughput.

For interactive or high-throughput deployments, benchmark smaller lookahead
values and confidence disabled. For this offline subtitle workflow, bounded
memory and continuous context are prioritized over minimum latency.
