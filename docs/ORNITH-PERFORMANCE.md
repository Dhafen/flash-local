# Ornith Performance Engine

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

Run **38954531421**, source `eafa2da636536c22a8dd4b4eaf3aeef3e95bb4bf`:
182 Swiftlet Core tests, 18 package tests, iOS Simulator compilation passed.
The two blocking lifecycle probe suites ran separately with only the
`.serialized` scheduling trait added in the validation harness. Original
assertions and deadlines were preserved. The mixed 4/8-bit synthetic graph
matched CPU output with maximum absolute logit error 2.2351742e-8 and the
same greedy choice. Twelve public tokenizer/template cases matched pinned
Python Transformers in both thinking modes.

These are small synthetic and metadata tests. Trained Ornith logits against
the publisher's reference dtype, sustained speed on iPhone, offline operation,
8K/32K/65K evaluated context, and coding/agent quality remain unverified.
Swiftlet uses F32 activations; stored BF16 weight validation alone does not
establish trained-model numerical parity.

The user's GGUF build 80 measured 1.493 decode forwards/s; build 84 measured
1.035 forwards/s, with 31 prompt tokens reused and approximately 304.5 MB
returned reads per decode forward. Prompts/history differ, so this is not an
isolated A/B result. No physical QPack speedup is claimed yet.

The MLX config mentions an MTP layer, but **none of the 1,757 published
tensors contains MTP weights**. MTP decoding cannot be enabled from that
field alone. Speculative decoding and the authenticated foreground LAN API
remain follow-ups after correctness and actual device comparison. Routing
top-k and the model graph are unchanged.
