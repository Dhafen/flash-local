# Ornith Performance Engine

See the [10–15 token/s work plan](ORNITH-10-15-TPS-PLAN.md) for the measured
build 94 **GGUF** baseline, code-level QPack bottlenecks, ordered experiments
and numerical/quality gates. The target is not yet established on iPhone.

A [separate grouped-expert prototype](https://github.com/Dhafen/flash-local-swiftlet/pull/2)
now passes 186 Core/lifecycle controls and iOS compilation in
[run 39027165387](https://github.com/Dhafen/flash-local-swiftlet/actions/runs/39027165387),
source `1e348e7b61289d70ba5bde2c212bcfa7867f4323`. Eight selected int4 expert
chains use three dispatches instead of 32. Actual Ornith projection shapes
with g64/BF16 scales match the original GPU output bitwise; small untrained
whole-model controls match logits/routes/KV, eviction and continuation.
This prototype is not integrated into the app; trained phone parity and
sustained A/B measurements remain required before activation.

The optional **Ornith QPack (testläge)** menu opens a separate chat and weight
directory. The existing GGUF/REAM runners, source weights and conversations
remain available. The app waits for one runner to release its model before
opening the other. The default chat stays on GGUF pending physical comparison.

## Reproducible source

- MLX repository: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`, revision
  `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`.
- Swiftlet upstream: `de1a942241f65ac43290cecffc85d69ea5b52310`.
- Runtime fork: `Dhafen/flash-local-swiftlet`, revision
  `1388b08883d3d469addb5fdb88916ac7c5a19cd6`. Changes omit the unused CPU
  KV mirror and instrument actual expert pread calls/returned bytes. The
  default upstream mirror behavior remains available.
- `scripts/inspect_ornith_mlx.py` fetches configuration and exact bounded
  safetensors headers only. It records 1,757 tensors in four shards, 40
  layers, 256 experts, top-8 routing, 30 recurrent/10 attention layers.
- Affine quantization: group 64, 4-bit experts, **8-bit routers and shared
  expert gates**, BF16 stored scales/biases. All are preserved byte for byte.
- Dense payload: **1,389,394,176 bytes**. Expert payload: **18,119,393,280
  bytes**. Original source files: **19,509,024,201 bytes** including headers.

## Installation and memory

`Experiments/OrnithPerformance` is a separate package. The installer writes
bounded 8 MiB HTTP ranges directly to QPack v1 dense/expert destinations in
private Application Support. It never writes a second raw checkpoint copy.
The source-bound journal supports interruption; restart hashing reconstructs
the source prefix from converted bytes. Full source LFS SHA-256 and a second
reconstruction from completed output both must pass before the runner opens.
Private model files are excluded from backups. Installation pauses in the
background; it resumes while the app is open. Existing GGUF files are not
deleted to make space.

The initial context capacity is 8,192, layer-major prefill chunks are at most
32 tokens, and the process ceiling is 5.5 decimal GB. The plan prices the
actual F32 GPU KV allocation, SSM/convolution state, growth allocations,
scratch, runtime and safety reserves before admitting approximately 2 GB of
expert cache. OS process headroom and Metal's recommended working set can
lower that allowance. Cache shrink happens only at an idle token boundary.

The app exports successful-forward wall time, GPU time when supported,
dispatch/command-buffer counts, cache counts, returned expert read bytes,
process footprint and thermal states. Returned pread bytes are not a storage
controller counter; concurrent read durations are not elapsed I/O wait.
iOS exposes thermal state rather than a public Celsius sensor. No prompt,
answer or token IDs are included in this performance export.

## Evidence and remaining gates

[Run **39020475546**](https://github.com/Dhafen/flash-local/actions/runs/39020475546),
runtime source `b8dd75d1e553111ab53655eeaf52ca2effc895d1`:
182 Swiftlet Core tests, 18 package tests, iOS Simulator compilation passed.
The two blocking lifecycle probe suites ran separately with only the
`.serialized` scheduling trait added in the validation harness. Original
assertions and deadlines were preserved. The mixed 4/8-bit synthetic graph
matched CPU output with maximum absolute logit error 2.2351742e-8 and the
same greedy choice. Twelve public tokenizer/template cases matched pinned
Python Transformers in both thinking modes.

The full app run also passed 572 app tests with zero failures and 474 package
test invocations with nine expected environment-gated skips and zero failures.
The close-before-open regression and exclusive model ownership checks passed.
Twenty exported iOS Simulator/macOS captures were visually reviewed in both
appearances. The device Release archive, Apple signature, memory entitlements
and privacy manifest passed inspection. Model payload is downloaded separately.

**Flash Local 0.1.0 (94)** is available in the existing private TestFlight
group. A separate read-only Apple audit on 2026-10-17 confirmed `VALID`,
`INTERNAL_ONLY`, `IN_BETA_TESTING`, nonexpired, and the exact existing group
and sole-owner membership. The signed IPA SHA-256 is
`d0243dc5de8e78678302112b9043f26319e62cd29257625cb668cae806647376`.
Update the existing installation, then open **ⓘ → Ornith QPack (testläge)**
and install its approximately 19.51 GB MLX checkpoint while the app is open.
This QPack checkpoint is separate from the existing GGUF file.

The numerical comparisons use small synthetic fixtures and public metadata.
Trained Ornith logits against
the publisher's reference dtype, sustained speed on iPhone, offline operation,
8K/32K/65K evaluated context, and coding/agent quality remain unverified.
Swiftlet uses F32 activations; stored BF16 weight validation alone does not
establish trained-model numerical parity.

The user's GGUF build 80 measured 1.493 decode forwards/s; build 84 measured
1.035 forwards/s. Build 94 measured 1.184 forwards/s over nine forwards,
31 reused prompt tokens, 22.023 s prefill for 21 newly evaluated tokens,
approximately 290.6 MB returned reads per decode forward and a 3.878 GB
observed footprint peak. Its model source remains the 21.7 GB GGUF with
4,096 context capacity, not QPack. Prompts/history and thermal conditions
are not controlled across these reports, so these are not isolated A/B
results. No physical QPack speedup is claimed yet. The MLX checkpoint uses
different quantization from GGUF Q4_K_M; byte-preserving QPack installation
alone cannot establish equal model quality across that change.

The MLX config mentions an MTP layer, but **none of the 1,757 published
tensors contains MTP weights**. MTP decoding cannot be enabled from that
field alone. Speculative decoding and the authenticated foreground LAN API
remain follow-ups after correctness and actual device comparison. Routing
top-k and the model graph are unchanged.
