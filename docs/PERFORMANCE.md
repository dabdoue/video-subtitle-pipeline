# Performance notes

These are focused development measurements, not general model benchmarks.
Provider version, PyTorch/Transformers build, GPU clocks, audio content,
lookahead, concurrency, and request format can change the result.

## Memory-aware offline policy validation

Measured on 2026-08-21 with RTX 3090 GPUs, PyTorch 2.13.0+cu130,
Transformers 5.15.0, fp16, and the same Nemotron 3.5 ASR checkpoint. The
candidate and original text-only server ran sequentially on the same spare GPU.
Wall time includes local HTTP upload/response handling. A 395.668-second real
Labware recording was used alongside its 10-second prefix and a synthetic
60-second repetition.

| Case | Original text-only | Offline + timestamps/confidence | Offline timestamps, no confidence | Stateful streaming + confidence |
|---|---:|---:|---:|---:|
| Warm 10 s | 0.278 s mean of 3 | 0.238 s mean of 3 | 0.201 s mean of 3 | 0.576 s steady mean of 2 |
| Warm 60 s | 1.990 / 1.272 s | 1.670 / 1.110 s | 0.955 / 0.923 s | 2.918 / 2.896 s |
| Full 395.668 s | 9.924 s | 8.545 s deployed | not measured | 22.113 s |
| Full peak | 1,796 MiB `nvidia-smi` | 5,986 MiB PyTorch reserved; 6,312 MiB `nvidia-smi` | not measured | 1,255 MiB PyTorch allocated |
| GPU after request | approximately 1,556 MiB | 1,588 MiB with cache release | not measured | approximately 1,556 MiB |

The first inference after switching execution mode or input shape can be
slower while CUDA kernels and algorithms warm up. For example, the first
10-second streaming call was 1.348 seconds before stabilizing near 0.576
seconds. Both first and repeated 60-second measurements are shown rather than
hiding that effect.

Findings:

- The deployed offline path was about 14% faster than the original text-only
  shim on the full recording even while returning word timestamps and
  confidence. Direct `AutoModelForRNNT` use also avoids some pipeline-wrapper
  overhead.
- Offline and streaming produced the exact same 1,023-word transcript on the
  full recording. Corresponding word timestamps differed by at most 80 ms
  (one encoder frame); mean absolute confidence difference was 0.00134 and the
  maximum was 0.03633.
- The original server independently reset context every 30 seconds. Its full
  transcript was 5,585 characters versus 5,565 for both continuous modes; no
  human ground truth was used here, so this is evidence of a difference, not a
  WER claim.
- Offline speed costs duration-dependent memory. The deployed policy estimates
  `256 MiB + 12 MiB/s`, predicting 5,004 MiB incremental for the full file.
  Observed `nvidia-smi` growth was about 4,756 MiB, so the estimate was roughly
  5% conservative on this sample.
- PyTorch's live allocated-tensor peak understated the CUDA caching allocator:
  3,177 MiB allocated versus 5,986 MiB reserved. `nvidia-smi` reached 6,312
  MiB including CUDA context/other process overhead. Memory policy therefore
  targets reserved VRAM, not only allocated tensors.
- With `NEMOTRON_RELEASE_OFFLINE_CACHE=true`, the full request returned GPU
  usage to 1,588 MiB. Without release, the first candidate retained about
  6,312 MiB after completion.
- The endpoint still serializes a single model instance. True batching of
  independent recordings/streams and progressive partial-response transport
  remain separate future work.

## Historical original versus streaming timestamp/confidence server

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

These measurements motivated the selectable policy above. Interactive use can
still benchmark smaller lookahead values, while offline subtitle work now uses
full-input decoding when the calibrated budget fits and bounded streaming when
it does not.
