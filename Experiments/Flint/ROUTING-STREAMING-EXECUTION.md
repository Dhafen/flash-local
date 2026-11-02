# Flint iPhone routing + streaming optimization — execution contract

## Goal
Optimize Recursive Flint 40B-A2.5B Q4_1+F32 (~25.99 GB) on iPhone 17 Pro Max without degrading numerical correctness. Target (not a promise): >=15 token/s decode and materially faster 1K/4K prefill. Current measured baseline: 1K prefill 9.47 token/s, decode 2.66 token/s, 201.81 s total, 684.57 GB requested expert reads; 128-slot sample 10.89/2.85; 256-slot sample 9.83/2.75. Those are single-run observations, not causal proof. Do not use requested read bytes as physical NAND traffic.

## Work autonomously, in stages; stop only on actual verification or a documented external blocker
1. Read and compare *actual code* and license/architecture of:
   - https://github.com/Edge0-AI/Edge0 (ios/Sources/Edge0Core/ExpertTensorStore.swift; macos/crates/engine-native/src/stepprefetch.cpp; docs/prerouter.md)
   - https://github.com/Helldez/BigMoeOnEdge (core/src/moe/router_hook.cpp; core/src/moe/expert_stream_source.cpp)
   - https://github.com/doramirdor/mbolt (layout, prefetch benchmarks)
   - https://github.com/Anemll/anemll-flash-llama.cpp (llama-moe-slot and Metal bank)
   Never copy code unless its license is compatible with this repository; preserve attribution.
2. Instrument the *pinned runtime's actual I/O path*: bytes physically submitted/read, read calls, latency/wait time, bank hits/misses/evictions, prefetch submitted/completed/useful/wasted, per-layer expert selection, GPU dispatch and synchronization times. Document exactly what each counter counts. No fake proxy metrics.
3. Implement bounded, cancellable asynchronous prefetch of the *exact selected experts* when routing IDs become available, using correct tensor/layer offsets and safe lifetime. Coalesce adjacent reads where possible. Do not mutate router logits, expert IDs, expert weights or top-k.
4. Introduce optional predictive prefetch only behind a separate flag. A prediction is advisory: if wrong, synchronously load the true routed experts and record the miss. Verify exact logits and routing parity vs baseline. Keep bounded memory, no unbounded threads, and ensure drain/cancel before unload.
5. Evaluate cache policies (LRU/LFU/hotness), per-layer vs shared slots, layout/repack feasibility, with held-out routing traces; never claim speedup from cache size alone.
6. Independently investigate n_ubatch=1 and disabled Flash Attention: the existing numerically admitted singleTokenF32KV mode is intentional. Enable faster modes only after native CPU/Metal full-model numerical parity at multiple positions and 1K/4K prompts. Fail closed on mismatch.
7. Measure real iPhone, cold and warm, baseline/each feature, >=3 repeats, thermal state, peak memory and memory pressure, 1K/4K prompt and >=128 generated tokens, identical settings. Preserve standard mode. No claims of 15-20 token/s without device data.
8. Run full Flint native tests, Swift tests, iOS compilation, and release archive validation. Keep TestFlight upload disabled until all required gates pass. Save run URLs and measured results in Experiments/Flint/evidence/.
9. Keep changes on a separate branch and open a draft PR. Iterate on actual CI failures. Never call a queued test passed. Document unverified aspects and any blockers.

## Architecture constraints
- iOS process memory budget 5.5 GB; current 8K context; 65K is NOT verified.
- GGUF Q4_1/F32, top-6 routing, 49-layer maximum, exact numerical routing and logits.
- No unsupported claim that expert read-ahead env flag is asynchronous routing prefetch.
- Retain conservative fallback, cancellation, safe model unload, and backward-compatible JSON.
- Do not silently change model output or enable lossy expert substitution.

## Acceptance
Complete only when correct native+Swift+iOS CI is green and real-device benchmarks demonstrate reproducible gains. If on-device measurements or signing cannot be performed from available tools, report BLOCKED ON DEVICE/UPLOAD explicitly, with exact completed commits and next required action; do not declare the performance target met.
