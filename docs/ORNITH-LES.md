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
The [LES ablation table](https://github.com/tc-mb/llama.cpp-omni/blob/96bbf61710d44421ea54e63f5194f9b17c20d877/docs/les/lookahead-expert-scheduling.md)+is explicitly for **iPad M4 16 GB**: 2.97 baseline, 9.81 scheduling/layout,
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
| A: existing packed GGUF/MLX | Phone build 94: 1.184 decode forwards/s, nine forwards; 3.878 GB peak | Same fixed prompt, at least 128 decode steps |
| B: llama.cpp-omni LES | Isolated source adapter and native control harness implemented; build/numerics pending | Whole-graph CPU/Metal, routing, reset, then phone |
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
versus lookahead execution, full logits, actual router decisions, greedy
choice and recurrent/KV reset. Trained Ornith parity, 128-token sustained
throughput, OS-cache effects, memory pressure, thermals and useful 8K/32K/65K
context remain device gates. An empty application expert cache is not a
guaranteed cold OS page cache. No winning backend is selected before those
measurements. The first speed milestone is 3 token/s, then 5 or 8 if proven.

DSpark needs a matching trained draft, target verification and recurrent
rollback. MiniCPM's draft is not an Ornith draft. Speculative decoding is
disabled; no quality/distribution equivalence is claimed. The foreground
LAN API remains a follow-up after correctness and decode speed.


The native streamed path requires Metal's alloc-only buffer type; CPU-only
streamed synthesis is explicitly unsupported by this upstream loader. Controls
use the complete resident tiny CPU model as oracle and compare resident Metal,
streamed Metal, lookahead and page-cache variants. This does not require a CPU
streaming implementation. A separate check compares the unchanged original
pinned CPU reference. The 256-expert/top-8 fixture repeats 128 teacher-forced
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


The hosted Apple Paravirtual GPU reports no SIMD-reduction capability, so the
unmodified feature gate rejects streamed Metal expert allocations before
inference. A CI-only path, restricted to macOS + Paravirtual + an explicit
flag, executes a 32-lane SIMD reduction and verifies all four known results
before allowing the reduction kernels. It does not enable SIMD matrix-multiply.
It is compiled out on iPhone. CI records this virtual-device test separately;
no A19 performance or hardware compatibility is inferred. Unavailable Metal
timestamps remain unavailable, rather than being reported as zero GPU work.
