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
  and the process-wide MLX allocator cache is capped at 32 MiB. This graph is
  not yet validated end-to-end against llama.cpp or enabled in the app.

On live iOS launches, the app registers its own Documents/Models root, creates
it on resolution, excludes it from backup and uses first-unlock file protection.
No picker grant is required for the app's own files. Scope bypass is restricted
to that explicit root; per-job bookmarks cannot escape it through a prefix
sibling or symlink. External/user-picked locations retain the existing grant
rules. Previews and injected storage do not gain implicit local authority.

These components do **not** yet constitute a validated model runner. Tokenizer,
complete attention integration/reference parity, complete PLE path,
runner binding to verified artifacts, downloader,
chat integration, app rebranding and signing still need implementation and
verification. No model is advertised as executable by these additions.

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

The Qwen35 tokenizer reads the GGUF's original byte vocabulary, token types
and ranked merge table through an explicitly bounded metadata opt-in. Layout
inspection still omits large arrays. Production tokenizer reads are tied to
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

## Important artifact discrepancy

The source HF config uses `qwen_sparse_attention`; the pinned converter only
marks `full_attention` for QSA. The published file therefore has all-zero
compression ratios, while its source config specifies ratio four. Compare
the public GGUF against that exact GGUF in llama.cpp before interpreting any
quality results. A repaired artifact requires a distinct manifest/hash and
separate reference validation. PLE layer IDs are intentionally converted from
HF's one-based indexing; that difference is not a defect.

No iPhone memory, token throughput, 65K context, full-model numerical agreement
or new TestFlight delivery has yet been established for this fork.
