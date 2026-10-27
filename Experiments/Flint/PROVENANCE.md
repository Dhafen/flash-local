# Flint engine source and format pins

This is an isolated port in the existing Flash Local repository. It does not
change Ornith/LES/QPack, app data, signing, existing workflows or app identity.
The root Swift package opts in only with `FLASH_LOCAL_ENABLE_FLINT=1`. The app
UI and existing LAN model remain unchanged pending Apple/device verification.

## Sources actually inspected

- [Flint model](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288):
  config, generation settings, tokenizer, model card.
- [Official INT4 GGUF](https://huggingface.co/RecursiveSI/flint-int4-gguf/tree/9c47151036bc2c34331142dda04488bceab2bc9f):
  complete GGUF header from a bounded 8 MiB prefix. Published full-file size
  and LFS SHA are in `model-profile.json`; the full payload is **not verified**.
- [Official MLX INT4](https://huggingface.co/RecursiveSI/flint-int4/tree/3b93f9053b530488405ae9fca51d4900d6ce21f0):
  metadata only; trained affine INT4, group size 64, INT8 embeddings.
- [llama.cpp Flint graph/converter](https://github.com/ggml-org/llama.cpp/commit/a37cc46104eb50900d5ab4a9ba02a0abdac039db)
  and [PR 30928](https://github.com/ggml-org/llama.cpp/pull/30928): graph, hparams,
  vocabulary regexes, stop-token handling, down-expert overflow scaling.
- [Pinned streaming runtime](https://github.com/tc-mb/llama.cpp-omni/tree/96bbf61710d44421ea54e63f5194f9b17c20d877):
  model loader, expert manifests, shared banks, slot remapping, preadv and graph.
- [Official Rec engine](https://github.com/recursive-oss/rec/tree/0e3150ca4262f572dee25c231d9ac7148414d2af):
  model equations, cache, engine, specialized kernels, checkpoint and chat.

`Native/FlintModel.cpp` derives from the MIT llama.cpp model graph; bounded
expert-bank plumbing derives from the MIT omni runtime. `FlintVocabulary.inc`
contains the exact upstream regex block. See `LICENSE.llama.cpp` and the pinned
runtime's included licenses (including minja's Google MIT notice).
`Native/FlintTemplate.jinja` is copied verbatim from Rec at the above revision;
the normalization behavior follows its `chat.py`. See `LICENSE.rec` (Apache-2.0).
The NumPy graph is independently expressed from the published model equations.

## Architecture and intentional restrictions

Flint is **not** Qwen's DeltaNet graph. It has 49 layers, two leading dense
MLPs, 47 MoE layers, 256 experts, sigmoid normalized top-6 and an always-active
shared MLP. Thirty-five attention layers have a 513-token window and RoPE;
fourteen full-attention layers have no RoPE. Q/K RMS normalization and learned
attention sinks are retained. No expert selections, stored quantization bytes,
RoPE settings or attention windows are changed to increase speed.

The first backend shares one expert staging allocation per projection/type/
stride across layers. It uses the OS page cache, no speculative expert choices,
no per-layer cache, no LES prediction and no DFlash. Actual top-6 IDs are
remapped to slots after routing. Source experts are read directly into writable
CPU/GPU-visible banks. The installed GGUF is retained via a verified descriptor;
only a small disposable offset manifest is written. No dense-weight derivative
or full second model copy is required by this port.

The fixed GPU weights use the runtime's owned shared-buffer type. This avoids
registering a giant Metal mapping spanning the interleaved expert gaps in the
26 GB GGUF. Whole-file mmap prefetch is disabled for Flint; CPU input embeddings
keep the existing file-backed row-lookup path. This plumbing is compiled in
the portable runtime but its actual Apple residency/footprint remains untested.

The official GGUF contains Q4_1 and F32; the loader initially rejects other
formats. Q4_1 is not Q4_K_M. The post-SwiGLU down-expert input uses upstream's
power-of-two scale with a 1024 headroom threshold, undone after the linear
projection. No nonlinear input or routing probabilities are scaled.

The older runtime needs explicit output graph expansion and Flint-specific
RoPE-pattern metadata. KV shifting is disabled, since globally shifting RoPE
would corrupt the no-RoPE layers. The Swift prefix policy only appends an
entire exact cached prefix; shortened/changed templates reset. It never rewinds
a compact SWA cache whose earlier cells may have been overwritten.

Native model lifetime is independent of a view. Cancellation and reset are
separate from unloading. The Swift actor serializes native operations; its
small locked cancellation bridge can be signaled during a synchronous native
call. The opt-in app integration now connects one actor to chat/API and enforces
a cross-runtime engine lease. The current serving configuration does not install
routing diagnostic callbacks. The first numerical comparison uses a separate
diagnostic context, then reloads the ordinary serving context after passing.
Apple build, visual and actual device checks are still required.

## Admission is not a phone benchmark

The inspected GGUF has 23,655,874,560 routed bytes and 2,331,622,528 other weight
bytes. Input embeddings alone occupy 1,229,979,648 bytes. Conservative admission
counts all fixed weights, padded F16 K/V, shared expert banks, current app
footprint and at least 1 GB scratch reserve. It does not budget a duplicate
full-KV checkpoint. Native checks and the Swift memory-pressure callback remain
necessary even when the estimate fits.

With an assumed 400 MB app footprint and physical batch 4, the current estimate
is 4.07 GB at 8K, 4.77 GB at 32K and 5.71 GB at 65K. The conservative 5.5 GB
admission therefore rejects 65K. These are calculations, **not measured RAM**.
Profiling mmap embedding residency and actual Metal scratch may allow a tighter
budget later. Metadata advertises 262,144 positions; that is not evidence of
usable iPhone context. No RoPE/window overrides are used to claim more context.

Rec's resident decoder requires a Mac with at least 48 GiB and specializes for
selected M4/M5 Pro/Max GPUs. Its desktop throughput is not an iPhone result and
its whole-model resident kernel cannot simply replace bounded phone streaming.

## Current validation boundary

See `evidence/cpu-controls.json` and `evidence/protocol-controls.json`. Small
untrained models verify the graph and streaming mechanics; the publisher's real
vocabulary verifies tokenization independently of weights. Prompt replay tests
verify format, not trained tool-call reliability. The Swift/Metal/iOS build,
actual trained logits, tool generation and sustained phone performance are
still unverified. Native sampling uses reusable GGML candidate storage; the
publisher defaults are temperature 1/top-p 0.95/top-k 20. Greedy controls are
separate. The native seed is reproducible within GGML, not asserted identical
to MLX random draws. No production model switch has been made.

The complete official GGUF SHA was verified on disk, without loading the model
into memory for inference. Twenty-seven real slices verify Q4_1 dequantization
and bounded CPU projections. The new source-identical Linux Swift controls test
tool schemas/replay, UTF-8/reasoning boundaries, prefix policy and comparison
rejection; they are not xcodebuild/Metal tests. Historical first-stage evidence
is retained; second-stage evidence states its own exact source/binary hashes.

An issue could not be opened before this larger adapter change because this
private repository has issues disabled. Repository settings were left intact;
work remains isolated on `codex/flint-engine` for review.
