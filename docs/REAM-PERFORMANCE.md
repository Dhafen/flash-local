# REAM performance work

Build 56 is a functional first path, not a performance-complete runtime. The
user reports extremely slow text generation on a physical iPhone. Their screen
shows 97.12 GB returned reader bytes, four generated tokens and 0.32 GB current
process footprint. These are a single screen observation, not a measured
decode rate, peak RAM or physical NAND traffic. The reader total also includes
prompt processing. No captured conversation is stored in this repository.

## Changes under validation

- Fused Metal GGUF dequantization/dot products preserve F32, F16, BF16, Q4_0,
  Q5_0, Q8_0, Q4_K and Q6_K layouts. The original CPU-dequantize/FP32-matmul
  route remains selectable for numerical and timing controls. Activations
  retain the pinned CPU's Q8_K/Q8_0/F16/BF16 rounding. No trained weight is
  requantized, and accumulators remain FP32.
- Routed experts share bounded packed dispatches: gate, up and down operate
  on the router's selected experts together. Each expert retains its own
  activation and weights. The canonical router accumulation order is kept.
- Prompt positions whose output is unused advance all KV/GDN/PLE states
  without running final HC or the vocabulary head. Only the last prompt
  position needs logits. This does not skip transformer layers or prompt tokens.
- A separate admitted dense-weight cache prevents cold experts from evicting
  frequently reused matrices. Whole eligible matrices are selected by size,
  then name, within a fixed payload capacity. All routed experts and the
  token/PLE tables remain streamed. Cache payload bytes, hits and reuse are
  reported; allocator overhead is not disguised as payload capacity.

The pinned 11,023,104-byte header (SHA-256
`352b87fd036c3b14c4bcb3ff83c19dc1802be26c93686b4300972e9a43390292`)
contains 43,902,566,400 expert bytes, 7,875,448,490 token/PLE table bytes and
4,326,571,520 other weight bytes. The Q8_0 output head alone is 675,430,400
bytes. These are tensor byte arithmetic, not measured I/O or useful speed.

The candidate app policy is 1,024 context, at most 5,500,000,000 total process
bytes, 4,250,000,000 admitted packed payload bytes and 64 MiB raw slots.
KV, recurrent state, a 1,000,000,000-byte working reserve and both caches must
fit the decoder's admission plan. Live footprint/OS-headroom checks remain
active. Cache contents are populated only on demand and disappear with the
generation engine. These limits are not evidence of an observed phone peak.
The original model file and its SHA-256 are unchanged, so updating the app
does not require a second model download. Do not uninstall the app.

## Reproducible validation and measurements

The regular hosted-macOS CI runs `GGUFPackedLinearTests` as part of
MLXBridgeTests, plus complete FP32/mixed-quant decoder and text-session
comparison with pinned llama.cpp. The tests independently check every
published encoding against canonical C dequantization and double-precision
dot products; distinct expert weights/activations; resident capacity and
identity validation; dense reuse after expert churn; final/following logits
after skipped prompt heads; and old measurement JSON decoding.

`testReproduciblePackedVersusFP32ProjectionBenchmark` prints
`GGUF_PACKED_PROJECTION_BENCH` JSON for Q4_K, Q5_0 and Q8_0, using deterministic
512 x 2,560 synthetic matrices, one warmup and three timing samples per path.
It records shape, source revision, OS, physical memory and byte counts.
The old timing includes CPU dequantization, copy and FP32 matmul; the new
timing is a resident packed Metal projection. Both omit file I/O and canonical
activation rounding. This separates the kernel/copy tradeoff from storage
and must not be presented as an end-to-end phone speedup. Debug test timings
also do not establish Release performance. CI runs no trained model inference.

Physical follow-up needs the same short input on the same model and phone,
time to first output, decode forward tokens/s, prompt/decode reader bytes,
cache reuse and observed peak footprint. First-use shader compilation, cold
versus warm weights and thermal state must be distinguished. Full-model
numerical comparison and offline proof remain outstanding.

## Other runtimes inspected

[Strata](https://github.com/Niko1221/Strata/tree/296b59d2f73c8f04a805df29c2fa97655adace78)
has a Metal engine as well as PC GPU paths. Its Mac documentation measures a
128 GB M5 Max and a 32 GB M4 Mac mini. It uses a large admitted expert arena,
packed quantized kernels, grouped work and optional contiguous expert files.
Its Mac speed figures are not iPhone evidence, and its residency/arena and
conversion assumptions cannot be copied into a 5.5 GB iOS budget unchanged.

[Swiftlet](https://github.com/leonickson1/Swiftlet/tree/de1a942241f65ac43290cecffc85d69ea5b52310)
explicitly targets iOS and streams Qwen3-Next/Qwen3.5/3.6 MoE experts with
Metal. Its authors report approximately 1 token/s for a 4-bit 35B on an iPhone
17 using about 2.5 GB RAM, and list a 42 GB 80B qpack alternative. These are
upstream claims, not measurements of this app or comparisons against REAM.
Both models have about 3B active parameters, reducing per-token work, but
switching requires a separate validated adapter/container and another large
download. Preserve and improve the already installed REAM artifact before
asking the user to download an alternative. No replacement model is selected
or downloaded by this performance change.
