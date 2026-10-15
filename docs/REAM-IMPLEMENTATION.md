# Flash Local REAM implementation

Own fork of Minirun; upstream base `3d477651e8180e1006928a66af8d611ff2286cdd`.
The original project and the existing Priv AI application are not modified.

## Current code

- `GGUFCore`: bounded v3 metadata parsing, exact per-type block geometry,
  checked tensor ranges, and one-expert/output-row descriptors for StorageCore.
- `CGGUFQuants`: allocation-free, tile-bounded decoding of Q4_0, Q5_0, Q8_0,
  Q4_K, Q6_K, F16, F32 and BF16. Quantized arithmetic is extracted unchanged
  from llama.cpp `e467e5b5e5ba43071d6151b3d881deb9300d000e`. See provenance
  hashes and full MIT license in that target.
- `GGUFStreamedLinear`: selected-expert row reads through Minirun's existing
  pool/leases, reusable FP32 scratch and eager MLX copy. Reports the extra
  copied tile explicitly. Evaluates each tile before proceeding. Output buffers
  are separately capped; this is not permission to allocate a complete model.
- Before non-F32 projections, `GGUFWeightSource` rounds activations through the
  pinned CPU dot kernel's Q8_K/Q8_0/F16/BF16 conversion functions. MLX still
  performs the FP32 product against unchanged, tile-dequantized weights. This
  matches the CPU reference's operand precision instead of merely increasing
  tolerance. Reusable output/packing scratch adds 384 KiB; temporary input and
  copied MLX activations are each capped at 256 KiB. No helper model/context or
  weight I/O is created. Projection reports include rounding calls/time.
- `Qwen4ExpConfig`: metadata-driven architecture and PLE head range validation.
  Published zero compression ratios remain dense; no implicit repair.
- `Qwen4ExpMemoryPlan`: overflow-checked KV/recurrent state admission under
  5,500,000,000 bytes, with mandatory caller-paid working memory reserve.
- `Qwen4ExpOperators`: V-first/modulo-head GDN step, causal/dilated convolution,
  sigmoid-gated norm, streamed-projection hyper-connections and top-k routing.
- `VerifiedGGUFArtifact`: full incremental SHA-256 against the pinned LFS
  manifest, no-follow rooted open, metadata bounds and a retained descriptor.
  Runtime readers receive duplicates of the hashed object. File identity,
  size and timestamps must remain unchanged before/after streamed operations.
  Header-only validation does not substitute for this full verification.
- `Qwen4ExpTextAttention`: one-token dense text attention, per-head Q/gate
  interleaving, head RMSNorm, partial RoPE and chunked FP16 KV storage.
  Online FP32 softmax avoids full-context KV expansion or a quadratic mask.
  Two chunked passes round the post-RoPE query and globally normalized
  probabilities to FP16, as required by the reference's FP16 K/V dot kernels.
  The second pass revisits the resident KV chunks, without additional weight
  reads. This trades additional score computation for reference arithmetic.
  Context exhaustion refuses instead of evicting. QSA and chunked prefill
  still require separate implementation and validation.
- `GGUFWeightSource`: connects the verification capability to a fixed raw LRU
  pool, reusable linear scratch, capped small tensors and one-row embedding/
  PLE reads. Per-tensor/expert/tile identities are stable across tokens. Full
  file integrity is checked before construction, and mutation stamps before
  and after each operation. Integration tests use synthetic verified files.
- `Qwen4ExpDecoder`: initial complete single-token numerical graph wiring
  embedding/PLE, HC, GDN/dense text attention, routed/shared FFN and LM head.
  Constructor validates all required shapes before opening weight readers.
  Failed/cancelled steps require reset. Live footprint checks retain headroom
  before the first weight read, each layer and routed expert. An unavailable
  footprint refuses the step and requires reset, including loss of telemetry
  after admission. Synthetic tests cover that refusal, reset/logit recovery,
  and zero additional reader bytes when budget or OS headroom rejects entry.
  The process-wide MLX allocator cache is capped at 32 MiB. This graph is
  validated against pinned llama.cpp with tiny random-weight graph/session
  fixtures. The iPhone chat route now uses it through verified GGUF authority;
  trained-model numerical parity remains unverified.

The synthetic reference harness also generates a separate 256-wide graph
with 113 random tensors (23.5 MB before quantization). Two recurrent key heads
and six value heads exercise the primary model's 1:3 modulo-head mapping.
Its test-only converter
uses the pinned `ggml_quantize_chunk` for Q4_K/Q5_0/Q6_K routed experts,
Q8_0 embeddings/shared projection, Q4_0 PLE, F16 HC, BF16 routers and retained
F32 operators. It requires all eight encodings in one file. A full five-position
CPU/streamed-MLX logit comparison uses the existing absolute 0.01 and relative
L2 0.001 limits; the corrected graph passed at `8ad2057` and `5b602d2`.
It does not establish trained-model numerical agreement or phone performance.
Run `38775973672` at `6c07d1a` reached all five mixed-graph positions but failed
the unchanged relative-L2 limit: 0.00374–0.00831 against 0.001 (maximum absolute
error 0.00120). The FP32 graph passed. The activation-rounding change above is
the resulting implementation fix, subsequently validated with the attention fix;
these pre-fix measurements are not presented as successful mixed inference.
Run `38779671123` at `80ce2a8` passed the focused operand-rounding tests. The
mixed graph's first position improved to relative L2 1.08e-7, but position one
still failed at 0.00167245; positions two through four were within the limit.
Delivery stopped before upload. A separate diagnostic replay now compares
bounded layer outputs and the pre-head activations, with quantization-boundary
observations. CPU and streamed diagnostic replays must stay within 1e-7 of
their normal logits; the unchanged normal runs remain the numerical oracle.
The streamed activation observer is DEBUG-only and is never enabled by the app.
Run `38783214287` at `aa4b39e` localized the mixed discrepancy to the final
dense-attention layer: the three recurrent-layer outputs stayed within
relative L2 1.2e-7 across all positions, while layer three reached 0.000274
at position one. The reference CPU's F16 K/V matrices also convert their RHS
(query and normalized softmax) to F16; the original MLX attention left these
operands in F32. The two-pass attention correction above is awaiting hosted
validation with the unchanged logit limits. The full-prefix scalar attention
test compares eight prefixes across chunk sizes 1/3/8, independently rounds
the operands, and emits bounded-cache/timing metadata. These tiny hosted
measurements do not establish long-context or iPhone performance.

Run `38785860218` at `8ad2057` passed the corrected attention checks and complete
FP32/mixed-quant graph gates at the original limits. Mixed maximum relative L2
was 2.969834632e-7 and maximum absolute error 4.4703484e-8 across five positions.
The 563 app tests and 472 package tests passed (nine environment-specific
package skips); iOS Simulator compilation also passed. The run was stopped
before any upload after a pre-upload audit found the missing required-reason
privacy manifest. This was not an Apple upload rejection. The manifest and
archive checks are now being added, with project regeneration and a new full
delivery run still required. These measurements remain tiny synthetic hosted
Mac evidence, not trained-model or physical-phone evidence.
The standalone CPU reference explicitly disables native tuning, Accelerate,
KleidiAI and llamafile, and selects the runtime's armv8-a target on Apple silicon;
its settings do not depend on a prior tokenizer build's CMake cache.

On live iOS launches, the app registers its own Documents/Models root, creates
it on resolution, excludes it from backup and uses first-unlock file protection.
No picker grant is required for the app's own files. Scope bypass is restricted
to that explicit root; per-job bookmarks cannot escape it through a prefix
sibling or symlink. External/user-picked locations retain the existing grant
rules. Previews and injected storage do not gain implicit local authority.

`GGUFModelTransfer` downloads exact HTTPS ranges through a body-bounded
URLSession delegate, records fsynced chunk hashes and the strong ETag, checks
committed chunks before resume, and publishes the final name only after full
SHA-256/GGUF verification. It makes no second full-size model copy. Installed
files are never overwritten. Core tests cover interruption, corrupt chunks,
changed identity, short responses, invalid headers and local storage limits.
A real URLSession test at `a294c6f` fetched only eight GGUF header bytes through
the pinned origin/CDN, preserving the ETag across the second If-Range request.

`REAMAppModel` and `REAMRootView` connect internal storage, verification,
download/resume and canonical generation on the normal iPhone launch. Workers
run hashing/I/O/MLX off the main actor; Stop keeps the UI busy until teardown.
Ordered bounded text events feed the transcript, and chat is saved locally.
Backgrounding stops the operation. The initial run policy is 1,024 context
tokens, 128 output tokens and 4,000,000,000 process bytes; this is a cautious
starting policy, not completion of the 65K goal. No fake output is substituted.

The actual app-controller session test at `a294c6f` passed exact prompt IDs,
eight generated IDs and decoded text against pinned CPU llama.cpp, persisted
the transcript, recovered after Stop and disabled Send when the file vanished.
That run failed separately because Git normalized the original model license's
CRLF bytes. `.gitattributes` now preserves the exact upstream bytes. New builds
are validating the fix. Earlier ImageRenderer chat captures omitted the native
scroll/input controls and are not accepted as visual proof; hosted-window
captures replace them and still require inspection.

The app has its own Flash Local identity and original model license available
offline. Device Release archives and a local automatic signing export are
optional manual CI gates. The export checks the actual signature/profile for
both memory entitlements and never uploads an IPA. Apple app-record setup and
TestFlight delivery remain unfinished; no real-model/iPhone result is claimed.

## Reproducible checks

Run `python3 scripts/fetch_ream_metadata.py` to obtain exactly 11,023,104 header
bytes from RemySkye revision `28a55fecdca51a498900aeb34a4e7e8338977cd3`.
The request checks HTTP 206, Content-Range, length and SHA-256; it never fetches
weight payload. Set `GGUF_HEADER_FIXTURE` to `.test-fixtures/ream.header`.
The published-header test checks all 1,224 tensors and byte totals, including
the last expert slice and compact PLE geometry. It skips explicitly without
that fixture; CI fetches it and rejects a skipped published-header test.

On Apple silicon, run the package suites with xcodebuild (not swift test):

```sh
python3 scripts/make_engram_test_fixture.py
xcodebuild test -scheme minirun-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation -skipMacroValidation \
  -only-testing:GGUFCoreTests -only-testing:MLXBridgeTests \
  -only-testing:ModelAdaptersTests -only-testing:StorageCoreTests \
  -only-testing:iOSBenchTests
```

The expanded StorageCore suite exposed an omitted upstream Engram fixture.
The independent Python fixture generator supplies 157 deterministic synthetic
rows in two v1 parts, including the partial final page. This restores executable
tests without claiming those bytes came from the unpublished model converter.

Numerical tests include manually constructed nonzero GGML quantization blocks,
independent matrix dot products with poisoned released slots, scalar golden
GDN outputs/state, exact PLE integer hash values, dilation/history boundaries,
HC group norms and routing ties. These tests are necessary suboperator checks,
not full-model logit comparisons. Actual model inference never runs on CI.

`GGUF_MATRIX_PROBE` records the same synthetic matrix run at 1/2/3 output rows
per tile: logical I/O, scratch and copied tile sizes, bounded output, wall,
I/O wait, decode/copy and GPU evaluation times. This is a reproducible tiny
regression probe; it is not a throughput prediction for the real model.
Process footprint, global MLX cache and allocator overhead remain the full
runner's responsibility and must be measured on the phone.

The full-graph harness generates a 193 KB untrained FP32 fixture with HC, PLE,
three GDN blocks, one dense attention block and routed/shared experts. Its
weights are deterministic random values, never copied from a checkpoint.
`scripts/run_synthetic_reference.py` builds the pinned CPU llama.cpp source and
records logits for fixed token IDs, including PLE EOS history. The decoder
parity test compares all logits over five positions with maxAbs <= 0.01 and
relative L2 <= 1e-3, then tests reset/cancellation. CI runs this separately from
suboperator tests and rejects skips. This is synthetic numerical testing, not
actual-model cloud inference or evidence of quality on the REAM weights.

The test-only Swift Qwen35 parity probe reads the GGUF's original byte vocabulary, token types
and ranked merge table through an explicitly bounded metadata opt-in. Layout
inspection still omits large arrays. Verified metadata reads are tied to
the fully hashed file descriptor. Qwen35 has no whole-word merge shortcut.
The typed text-chat path binds the exact published template hash and selects
`enable_thinking=false`, `preserve_thinking=false`, with no tools or images.
Its trimming follows the pinned llama.cpp Jinja implementation's C-locale
whitespace behavior; it does not silently normalize Unicode.

The vocabulary-only reference fixture copies the hash-checked header's KV
bytes into a GGUF with zero tensors. The CPU reference opens it with
`vocab_only=true` and never creates an inference context. It supplies exact
token IDs and decoded bytes for 109 public strings, plus Jinja rendering and
token IDs for 22 text conversations. The parity gate must pass before this
new tokenizer/chat path is considered validated. No user conversations or
trained inference enter the cloud reference harness.

The product tokenizer must reuse canonical llama.cpp/Jinja, as required by the
approved plan. `Sources/CLlamaTokenizer` is the C ABI for that path: it loads
`vocab_only=true` from a duplicate of the verified descriptor, owns no tensor
payload, delegates tokenization, token pieces and EOG to llama.cpp, and runs the
unchanged Jinja engine for typed no-thinking text chat. The reference harness
also checks the bridge's buffers, token/byte equality, EOG IDs, descriptor
ownership and invalid requests. `build_tokenizer_xcframework.py` builds arm64
Mac/device/Simulator slices from the same source pin before Swift package
resolution. `CanonicalQwen35Tokenizer` binds the C API to verified GGUF
capabilities; its tests compare all public strings/conversations and reject
file mutation after vocabulary loading. The experimental Swift probe is not
registered as a runner and lives under `Tests/ModelAdaptersTests/Support`,
outside the product source targets. Canonical tokenizer and complete synthetic
generation gates passed in run `38709619293` at `6b44062`.

`Qwen4ExpTextGeneration` connects canonical text prompts to the streamed
decoder, with deterministic greedy sampling, EOG handling, byte-correct UTF-8
streaming, cancellation, explicit context/output limits and I/O/physical-memory
telemetry. It distinguishes sampled run footprint from process-lifetime peak.
A separate 237 KB random-weight fixture adds a tiny byte vocabulary to the full
test graph; the CPU reference and Swift engine must produce identical prompt
and generated IDs, then repeat after cancellation. This is not trained-model
quality or iPhone performance evidence. The product iPhone route is implemented;
build `0.1.0 (56)` passed compilation and hosted visual review in light/dark on
macOS and iOS Simulator. The captures label preview text explicitly; they are
not physical-phone inference evidence.

The app's short numerical diagnostic saves the CPU/MLX logit comparisons with
model hash, source/reference revisions, exact token IDs, measured memory/I/O,
and the numerical limits. Its versioned JSON also records a timestamp, OS,
physical memory and hardware class, explicitly distinguishing `iOSDevice`,
`iOSSimulator` and `macOS`. No hostname or unique device identifier is collected.
Agreement applies only to the recorded positions; the environment stamp does
not establish airplane mode, long-context correctness or model quality.

## Important artifact discrepancy

The source HF config uses `qwen_sparse_attention`; the pinned converter only
marks `full_attention` for QSA. The published file therefore has all-zero
compression ratios, while its source config specifies ratio four. Compare
the public GGUF against that exact GGUF in llama.cpp before interpreting any
quality results. A repaired artifact requires a distinct manifest/hash and
separate reference validation. PLE layer IDs are intentionally converted from
HF's one-based indexing; that difference is not a defect.

TestFlight `0.1.0 (56)` from app source `5b602d2` is now confirmed `VALID`,
`INTERNAL_ONLY` and `IN_BETA_TESTING` for the sole configured private group.
Run `38791263363` passed all preceding gates and uploaded successfully, then
failed its final unsupported beta-group GET. The corrected API helper in
`f78d164` completed and verified association against that existing build.
The CI conclusion remains failure; recovery is recorded separately. No
physical-phone installation, memory, throughput, full-model numerical
agreement, offline inference, 65K context or quality comparison is established.
