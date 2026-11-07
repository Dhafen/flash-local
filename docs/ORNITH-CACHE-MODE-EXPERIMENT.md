# Ornith: iPhone expert cache vs. OS page cache

## Goal

Increase **generated model tokens/second** (decode), ideally reaching 15 t/s
on a physical iPhone 17 Pro Max, **without altering Ornith GGUF weights,
quantization, router/top-k, tokenizer, or generation settings**.

This change does not claim 15 t/s. It provides a bounded alternate
weight-residency path, a real-phone measurement, and a conservative switch.
Flash/NAND IOPS or transactions/second are **not** LLM tokens/second.

## Modes

- `omniPageCache` (default): the existing OS file page cache; no explicit
  native per-layer expert window.
- `boundedExpertCache` (optional): same GGUF and graph, but the native
  streaming runtime retains a memory-admitted expert window. The plan must
  hold at least the eight selected experts per layer. No full-model fallback.

Both are admitted under the existing 5.5 GB process ceiling and its independent
headroom/footprint checks. Larger context or insufficient RAM can refuse the
alternate mode; it must not silently change quantization or truncate tokens.
Until a combination is independently proven, the alternate mode uses the
verified default physical/evaluation batch, not a profile established for
OS-page-cache execution. Switching modes invalidates and rechecks saved
KV/SSM prefix restoration admission.

## On-device benchmark

In Ornith quick chat, open **Kontrollera KV- och prefixcache**, then
**Jämför expertcache mot sidcache**. The real installed GGUF must pass its
original-weight numerical check before this experiment.

Four new owners execute in this order:

1. OS page cache
2. Bounded native expert cache
3. Bounded native expert cache
4. OS page cache

Each sends the same original-model user prompt, greedy sampling, enabled
thinking, and at most 129 generated tokens. Startup/model initialization is
outside the timed generation. The report records prompt token IDs, generated
token IDs, prompt/decode timings, I/O, memory plans, and thermal states.

**No winner** if any run ends before 129 generated tokens / 128 decode forwards,
the token ID sequence or prompt changes, any response reuses an old prompt
prefix, the device is in Low Power Mode or serious/critical thermal state,
thermal state changes within a run, or footprint exceeds the admitted plan.
Candidate must actually allocate >=8 expert cache slots per layer; baseline
must allocate 0. The OS file cache cannot be forced cold, so ABBA reduces
order bias but does **not** isolate NAND traffic.

If comparable, the app offers a **manual** switch only when the expert-cache
candidate gains at least 10% in decode speed **and** retains at least 85%
of the baseline prefill rate. Otherwise it keeps OS page cache.

The option is **session-only**. The report is exported to the user's local
Documents/Diagnostics folder. The UI offers **Återgå till sidcache**.

## What is still unverified

- Physical iPhone 17 Pro Max speed, 15 t/s, 30-minute thermal stability.
- Full trained-model numerical equivalence beyond the admitted original-
  weight controls and the 129-token ABBA identity.
- 65K context accuracy/footprint, background operation while iOS suspends an
  app, and any TestFlight delivery.
- Larger speculative/DFlash decoding or Metal dispatch fusion.

The production default remains unchanged until phone measurements support an
alternative. Inspect decode vs prefill and cache-hit/physical-read counters
separately: faster file reads alone do not establish faster token generation.

## Review and CI

Follow repository DCO policy and test with `xcodebuild` (not `swift test`).
The portable policy test is
`MinirunRunnersTests/OrnithLESChatTests` and rejects missing/nonfinite/
insufficient benchmark improvements. Run the integrated iOS Simulator build
and real-device ABBA before treating this as a performance release.

## Demand I/O in the live Ornith runtime

The original GGUF reader groups at most eight adjacent demanded experts into
one vectored read per tensor family. The buffers are the actual existing
Metal-visible slots. It reads no bytes for gaps and never changes routing,
quantization or computation order. A context reuses its I/O workers across
layers and waits for every write before exposing the populated slots.

For demand-only expert cache, the native loader checks that all decode views
are disjoint from the prefill workspace. The layer can then retain its LRU
metadata through a prefill wave and reuse still-resident experts in the
next decode. No expert tensors are added or duplicated. The existing
memory admission and manual on-device comparison still apply.
The prepared GGUF format is compatible with build 428. The app verifies its
existing files and uses a fresh runtime admission for the updated engine.

### Recorded Ornith 35B trace

The Phase 2 SparseInferenceLab trace contains 33,600 routes from six complete
public tasks. Its reconstructed JSONL SHA-256 is
`2909c82f9a29a1b437018f281111dccb0be75c55819815dda6943593a58438d5`. Replay uses the six
tasks in sequence as requests sharing one context. With physical batch four
and a 25-slot decode window, simulated decode demand drops from 34.55 GB to
33.10 GB (4.21%). Logical decode spans fall from 54,339 to 51,219 (5.74%).
In page-cache mode the byte ledger is identical and the number of prefill
spans falls from 489,063 to 446,289 (8.75%). These are modeled file requests.
The operating system page cache and physical storage are outside this replay.

```sh
# routes.jsonl is reconstructed from the experiment's sorted .jsonl.gz parts.
python3 scripts/measure_ornith_demand_trace.py routes.jsonl --physical 4 --slots 25
python3 scripts/measure_ornith_demand_trace.py routes.jsonl --physical 4 --slots 0
c++ -std=c++17 -O2 -pthread Experiments/OrnithLES/Native/expert-io-control.cpp -o /tmp/ornith-expert-io-control
/tmp/ornith-expert-io-control --benchmark
```

The C++ benchmark is a warm-file host microbenchmark with deliberately
adjacent requests. It checks both paths' destination bytes after the same
waves. The Apple workflow separately tests the native Ornith graph against
the established reference and checks that a repeated synthetic decode
following prefill reuses the exact weights without reading them again.
Physical iPhone TTFT, prefill, decode, memory and sustained thermals require
a device run. The replay neither measures nor implies a 15 tokens/s result.
