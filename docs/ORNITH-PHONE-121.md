# Ornith phone benchmark: build 121

User-supplied physical iPhone18,2 report, app source `a30fe6eefa4f-dirty`,
2026-10-18 19:41:36 UTC. Report SHA256:
`9356e8f93f453938aff60a9bbee56b395f298bea36026eb5bfcf67c61b75e013`.
The five subsequently shared files repeat this report and four build-120
exports; they are not independent repetitions.

Model: the installed Ornith 1.5 35B-A3B Q4_K_M, 21,713,463,040 bytes,
SHA256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
No conversion, additional model download, routing change or speculative
decoding was used. The capacity is 4,096 tokens, not an exercised 4K prompt.
Each run has the same 60-token benchmark prompt and 128 generated tokens,
with 128 completed timed decode forwards. Low Power Mode is off. Airplane
mode is user-declared, not independently verified. All eight runs start and
remain at iOS thermal state 1 (fair); temperature in degrees is unavailable.

| Backend | Initially empty app cache: decode tok/s | Warm app cache: decode tok/s | Prompt tok/s, first / warm |
| --- | ---: | ---: | ---: |
| Existing GGUF | 1.192 | 1.178 | 1.130 / 1.196 |
| Native expert cache | 3.278 | 3.703 | 3.492 / 4.019 |
| Native LES, depth 4 | 4.087 | 4.156 | 4.030 / 4.364 |
| Native OS page cache | 5.015 | 4.850 | 4.993 / 4.533 |

The three native modes have identical 128-token output digests in all six
runs. OS page cache is the fastest of these comparable native modes in both
runs. The existing engine has a different output digest. Its observed rate
is retained, but an output-equivalent four-way speedup is not established.
`scripts/analyze_ornith_les.py` preserves that failed comparison and reports
the complete native-only comparison separately. It never selects a production
backend automatically.

Run order is fixed and OS-cache coldness is uncontrolled. Initially empty
application caches do not establish cold physical flash. These are two runs
per mode on one fixed prompt, not a sustained workload or a quality suite.

## Numerical admission

The original quantized-operand CPU comparison still differs by 3.1191 maximum
absolute logit error / 0.06596 relative L2. Its original failed status is kept.
The independent original-weight F32-activation/double-accumulation reference
has serial errors 0.00377691 absolute / 0.0000444115 relative L2, within the
unchanged 0.01 / 0.001 limits. Four full-vocabulary positions have the same
greedy choice. Callback-free cache, LES and page-cache executions pass those
same controls.

All 8,000 position/layer/expert choices agree as exact top-eight sets across
25 forwards. Reference rank order differs within those sets; native LES
on/off ranked traces remain identical. This does not change real selections.
Batch-four still fails (2.28847 absolute / 0.0592567 relative L2); all native
timings therefore use independently admitted single-token prefill.

Four positions do not establish general model quality or long-context
correctness. `fullModelNumericsVerified` remains false. A separate experimental
chat can require a fresh in-process admission; the ordinary engine is retained.

## I/O and memory scope

Decode returned-read bytes per forward are 270.61 MB for the existing engine,
255.71 MB for the native cache, 282.43–284.43 MB for LES and 609.48 MB for the
page-cache mode. The latter totals 78.01 GB per 128-forward run. These are
cumulative returned read bytes, including repeated reads of the same experts,
not unique file size or measured physical flash traffic. OS-cache hits still
count. Dense file-mapping faults are not included in the reader counter.

Sampled process footprints are approximately 3.81–3.82 GB for the existing
engine, 2.25–2.29 GB for the native cache/LES and 0.375 GB for page-cache mode.
The small page-cache footprint is **not total system RAM consumed by inference**:
reclaimable file cache and file-backed residency are not fully represented by
the process footprint. Native admission also reserves dense/state/staging
memory and monitors current iOS headroom against the 5.5 GB process ceiling.

Page-cache decode has 237,184 Metal dispatches and 10,624 timestamped command
buffers over 128 forwards; the timestamped GPU sum is 7.89–8.05 seconds.
The remaining wall time includes scheduling, I/O, synchronisation and CPU
work. GPU sums do not independently isolate each of those costs. This evidence
supports evaluating the page-cache path first; it does not promise 10–15 tok/s.

Private raw reports and conversation/token payloads are not committed.
