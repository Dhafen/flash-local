# Ornith LES experiment

This isolated experiment reuses the installed Ornith Q4_K_M: 21,713,463,040
bytes, SHA-256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
Production GGUF/REAM and Swiftlet/QPack paths are unchanged. Native controls
must pass before app integration; no phone throughput is claimed.

## Source inspection

Reviewed source snapshots:

- OpenBMB/MiniCPM-V-Apps `fa9d837e3d38dfde7a9484ec1cc6bc2e5bc9b420`:
  `MiniCPMModelConst.swift`, `MBMtmd.mm`, build scripts and submodule pin.
- tc-mb/llama.cpp-omni **ios-omni** `96bbf61710d44421ea54e63f5194f9b17c20d877`:
  `src/models/qwen35moe.cpp`, `llama-moe-slot.cpp`, `llama-moe-experts.cpp`,
  Metal implementation, `tools/moe/experts.py` and `strip_experts.py`.
- Anemll/Flash-iOS `4f1a69777629cda07b4fdd7d93511ed43f62fad5`:
  `FlashMoEEngine.m`, `metal_infer/infer.m`, shaders and I/O experiments.
  This snapshot has no license grant file; its code is not copied or shipped.
- Swiftlet's QPack/Engine/Metal sources upstream `de1a942241f65ac43290cecffc85d69ea5b52310` and our separately tested grouped
  dispatch fork, described in [Ornith Performance](ORNITH-PERFORMANCE.md).

The [publisher's 13+ token/s iPhone claim](https://github.com/OpenBMB/MiniCPM-V)
does not supply an exact reproducible prompt/run/thermal/configuration record.
The [LES ablation table](https://github.com/tc-mb/llama.cpp-omni/blob/96bbf61710d44421ea54e63f5194f9b17c20d877/docs/les/lookahead-expert-scheduling.md)
is explicitly for **iPad M4 16 GB**: 2.97 baseline, 9.81 scheduling/layout,
14.92 with mixed quantization, 17.39 with DSpark. These are publisher results,
not Ornith measurements. The current app defaults to Q3_K experts, top-8,
4096 context, lookahead depth 8, mmap dense weights and DSpark **off**.
Its 12 GB device memory ceiling is 6 GiB, exceeding our 5.5 decimal GB limit.
The 3 GB marketing figure is not evidence of total process peak footprint.

## Compatibility and retained mathematics

Both trunks use 40 main layers, 256 experts/top-8, 30 gated DeltaNet layers
and ten full-attention layers. The loader reads dimensions, RoPE sections,
frequency, RMS epsilon, KV heads and SSM geometry from the GGUF. Ornith also
stores an unused MTP block; main execution excludes it. A local patch avoids
predicting that unused block's experts. No routing override is enabled.
Tokenization/chat formatting continues to use the existing canonical bridge
and the original file's exact metadata. MiniCPM's vision tower is not loaded.

`experts.py` copies quantized family slices into contiguous expert-major
layer files; `strip_experts.py` retains metadata and raw dense tensor bytes.
Neither inherently requires requantization. Repacking all experts would
duplicate roughly the full installed model while preserving rollback.

Our first variant therefore copies/readback-verifies only the dense tensors
with 4 MiB buffers, preserving all raw GGUF metadata and tensor types. It
keeps expert tensors in the verified original descriptor. The local
`gguf_tensor_major` extension reads each gate/up/down span directly into its
GPU-visible destination. This costs up to three source calls per expert
instead of a contiguous preadv; device comparison must decide if repacking
is worth its storage cost. No full-model Metal mapping is created.

LES predicts the next layer from its router and the current normalized
activation. Predictions guide asynchronous prefetch; actual top-k still
determines computation, including all missed predictions. Slots remain
protected from reuse until the corresponding graph operation permits it.
The runtime uses per-layer decode windows plus shared prefill banks grouped
by tensor family and quantization stride, including mixed Q4_K/Q6_K down
projections. All buffers, prefetch slots and temporary prefill banks count
toward admission. No static MiniCPM expert-frequency/prewarm table is reused.

## Candidates and acceptance

| Candidate | Current evidence | Next decision |
| --- | --- | --- |
| A: existing packed GGUF/MLX | Phone build 107: 1.931 decode forwards/s, 190 forwards; 3.954 GB peak; uncontrolled prompt/cache | Same fixed prompt, at least 128 decode steps per mode |
| B: llama.cpp-omni LES | Build 107 host/reference gates pass; first phone model load fails before numerical check | Resolve native loading, then trained Metal/reference gate and phone comparison |
| C: Flash-MoE pipeline | Code inspected; fused expert/pointwise kernels, deferred command buffers, parallel pread; no redistribution license found | Do not copy; compare permitted independently implemented techniques |
| Swiftlet/QPack | Build 97 delivered, grouped kernels conditionally activated after four-position replay; no phone result | Different checkpoint quantization, so cannot be an identical-weight A/B |

The dynamic LES framework exports only `flo_*`; its llama/GGML symbols must
not replace the production tokenizer/reference runtime. CI verifies that
isolation for macOS, iPhone and Simulator. Aggregate instrumentation counts
all expert reads including prefetch, actual Metal dispatch calls, command
buffers and GPU command-buffer timestamp durations. GPU duration is not
wall latency; overlapping durations can sum beyond elapsed time. Application
read bytes are not physical flash-controller bytes.

Initial tests use tiny untrained fixtures only: resident versus streamed
versus lookahead CPU execution, full logits, actual router decisions, greedy
choice and recurrent/KV reset. Trained Ornith parity, 128-token sustained
throughput, OS-cache effects, memory pressure, thermals and useful 8K/32K/65K
context remain device gates. An empty application expert cache is not a
guaranteed cold OS page cache. No winning backend is selected before those
measurements. The first speed milestone is 3 token/s, then 5 or 8 if proven.

The existing backend requests `F_NOCACHE=1` on a duplicate of the verified
descriptor. Native trials explicitly apply `F_NOCACHE=0`, including when the
descriptor inherits that flag from an earlier reader. Failure to apply the
policy stops the native trial. Native cache and LES modes use both the OS
page cache and their own windows; the OS-only mode drops the decode windows.
This first comparison does not isolate a native own-cache-only policy. All
reported read bytes are explicit source reads; dense mmap page faults are
not included. Actual physical flash traffic is not exposed by these counters.

DSpark needs a matching trained draft, target verification and recurrent
rollback. MiniCPM's draft is not an Ornith draft. Speculative decoding is
disabled; no quality/distribution equivalence is claimed. The foreground
LAN API remains a follow-up after correctness and decode speed.


The iPhone streamed path requires Metal's alloc-only buffer type. Host controls
use a bounded CPU allocation adapter, while production continues to require
Metal. A separate check compares the unchanged original pinned CPU reference.
The 256-expert/top-8 fixture repeats 128 teacher-forced
positions to exercise eviction and prefetch races without trained weights.

The opt-in app benchmark uses the existing verified file capability, prepares
only dense weights, performs a bounded CPU-reference comparison, then compares
four modes over two genuine greedy generations each. Early EOS is retained
and marks that run shorter than the required 128 tokens. Batched prefill has
its own reference gate. Numerical disagreement prevents candidate benchmarks;
the original chat remains available. The default production package does not
link LES unless `FLASH_LOCAL_ENABLE_LES=1`. The dedicated delivery input requires
the native/reference/Swift gates and transfers an exact-source/hash-checked,
symbol-isolated XCFramework before app/archive/signing gates. No device result
or TestFlight delivery is established by implementing those gates.


The hosted Apple Paravirtual GPU advertises no SIMD reduction. An explicit
instruction probe also failed; it was removed rather than forcing a capability
flag. Numerical CI therefore uses a complete resident CPU oracle versus the
same streamed/cache/lookahead/page-cache graph with host allocations. That
host allocation adapter is macOS-only, requires an explicit control flag,
disables mmap for the fixture, and is restricted by the C API to at most
32 MiB. The Swift test additionally requires the generated fixture name.
It cannot enable CPU streaming on iPhone. No mathematical operator or routing
is changed. Metal is compiled for all three platforms but its runtime numerics
remain a physical-device gate. The app always uses Metal and must compare its
trained logits with the existing CPU reference before benchmarking.

## Build 107 phone evidence

The user-provided ordinary GGUF export reports 190 decode forwards in
98.381 seconds (1.931 forwards/s), 28,962,661,120 returned read bytes during
decode (152.435 MB/forward), and 3,953,939,248 bytes sampled peak footprint.
Prefill evaluates 26 new tokens while reusing 270: 17.701 seconds and
6,023,248,128 returned read bytes. Total source reads for this generation are
34,985,909,248 bytes. The screenshot shows a different, longer prefill with
506 of 511 prompt tokens completed and 90.09 GB returned reads.

The read counter is a cumulative per-generation sum of completed source
reads, including repeated extents and prefetched bytes. It is neither unique
file coverage, newly downloaded/storage bytes nor simultaneously resident
memory. Repeated expert eviction and token-at-a-time prefill can make it much
larger than the 21,713,463,040-byte file. Explicit reads can also be served by
the OS cache; these counters cannot establish physical flash traffic. The
existing chat backend keeps approximately 1.39 GB fixed weights plus 2.11 GB
evictable weights within its 3.5 GB packed-weight capacity. Further memory
holds activations, KV/SSM state, runtime and UI. Increasing history can require
many repeated reads even when the visible user message is short.

The separate LES export has zero measurements and reports `LES model loading
failed`, with a 356 MB observed footprint. Thus the above throughput belongs
to the existing engine. No LES speed, trained-model parity or fastest-backend
claim follows from these reports.

The loader investigation found another pathname operation on the verified
descriptor: manifest extent validation used `filesystem::file_size` on
`/dev/fd/N`, although expert reads already retained the descriptor directly.
The candidate fix uses `fstat` for extent validation and `F_DUPFD_CLOEXEC`
for retention, without traversing `/dev/fd`. This addresses a possible iOS
sandbox restriction; the generic build-107 report does not prove it caused
the observed failure. Bounded error-only native diagnostics now preserve the
loader's actual cause with local paths removed. Descriptor parsing/retention,
diagnostic limits/privacy and an invalid-header C-API regression join the
native controls. Physical confirmation remains required. No weights, router
selection, quantization or production GGUF/REAM computation are changed.
