# Ornith phone benchmark: build 123

User-supplied physical iPhone18,2 report, source `4d3f45e3520c-dirty`,
2026-10-19 06:08:43 UTC. Report SHA256:
`65211c7c7e1118482b63556baa92f3ea4179d14063dcc4d681872bf9a58abf57`.
The reported source matches the [delivered build](TESTFLIGHT.md); this is not
independent attestation of the exported file or airplane mode.

The installed Ornith 1.5 35B-A3B Q4_K_M hash is unchanged:
`42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
All eight runs have the same 60-token prompt, 128 generated tokens and 128
completed timed decode forwards. Capacity remains 4,096 tokens. Low Power
Mode is off; all runs start and remain at nominal iOS thermal state (0).
Application-cache labels do not control the OS page cache or fixed run order.

| Backend | Decode tok/s, first / warm | Prompt tok/s, first / warm | Prefill batch |
| --- | ---: | ---: | ---: |
| Existing GGUF | 1.156 / 1.151 | 1.102 / 1.057 | 1 |
| Native expert cache | 3.675 / 3.664 | 5.176 / 5.864 | 4 |
| Native LES depth 4 | 4.044 / 3.834 | 5.833 / 5.306 | 4 |
| Native OS page cache | 4.299 / 4.082 | 7.253 / 7.189 | 4 |

Page cache wins both comparisons among native modes. All six native outputs
are identical and match their build-121 output digest. The existing engine
still has a different output digest; an output-equivalent four-way speedup
is not established. This remains one prompt, two runs per mode.

## Batch-four now passes on the trained model

The independent original-weight F32-activation/double-accumulation reference
admits serial and batched execution with unchanged limits (absolute 0.01,
relative L2 0.001). Serial errors are 0.00377691 / 0.0000444115; batch-four
errors are 0.00388515 / 0.0000451105. All four full-vocabulary positions have
the same greedy choice. The callback-free cache/LES/page-cache controls pass.
All 8,000 choices across 25 forwards agree as exact top-eight expert sets;
rank order within those sets can differ from the independent reference.

This closes the previously failing phone batch-four control following the
[Q4_K intermediate-precision correction](ORNITH-LES.md). No tensor bytes,
quantization or actual top-eight selections changed. The old quantized-operand
CPU comparison still fails (approximately 3.119 absolute / 0.06596 relative L2)
and is preserved. Four positions do not establish general model quality:
`fullModelNumericsVerified` remains false.

Page-cache prefill takes 8.273 / 8.346 seconds for 60 tokens. In build 121,
serial prefill measured 4.993 / 4.533 prompt tok/s. Decode in this report is
slower than build 121's 5.015 / 4.850 tok/s. Different thermal states, OS-cache
conditions and test sessions prevent attributing that change to one patch.
The new evidence establishes batched correctness and faster observed prefill,
not a new decode-speed record.

## Remaining decode cost

Page-cache decode averages 233 / 245 milliseconds per forward. Expert install
spans average 128 / 133 ms and timestamped Metal sums average 74 / 80 ms.
These spans can overlap and must not be added as exclusive costs. There are
1,853 Metal dispatches, 83 command buffers and 960 returned read calls per
forward. The pinned source also creates and joins I/O threads per layer wave;
a bounded persistent worker pool and thread-count sweep are candidates to
measure, not implemented or measured improvements in build 123.

Page-cache returned reads are 609.48 MB per decode forward, including repeated
expert reads and OS-cache hits. This is not unique model size or measured
physical flash traffic; dense mmap faults are omitted. Sampled process
footprint is about 0.487 GB for page cache, 2.38-2.39 GB for native own-cache
modes and 3.82 GB for the existing engine. File-cache RAM is not fully included
in process footprint, so these values do not establish total inference RAM.

A separately supplied quick-chat screenshot shows 133 generated tokens,
4.50 token/s, 2.33 seconds of prefill and a collapsed reasoning section.
It confirms the UI reached a completed answer, but supplies no raw timing,
prefix-reuse or logit trace beyond those displayed values. It is not a ninth
controlled benchmark run. Raw reports, screenshots and token payloads remain
private. Long-context, general quality and sustained workload tests remain open.
