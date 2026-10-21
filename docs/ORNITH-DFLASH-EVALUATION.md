# Ornith DFlash evaluation

Source inspection: 2026-10-21. **No draft weights downloaded, no DFlash model
inference executed and no phone speedup measured. DFlash is not enabled.**
The installed 21,713,463,040-byte Q4_K_M GGUF and existing app remain unchanged.
Target SHA256: `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.

## Decision

A native Metal experiment is technically credible. Prefer the existing
llama.cpp-omni runtime and expert streamer, with a separate dense draft context.
Do not replace Flash Local with a Python/vLLM service or assume that adding a
0.4B file is sufficient. Prioritize the pending KV/lifecycle delivery first.
65K plus a BF16 DFlash draft does **not** fit the current conservative memory
plan without further work; that is a budget calculation, not a hardware failure.

## What the current app actually does

- `Experiments/OrnithLES/Native/FlashOrnithLES.cpp` uses
  [llama.cpp-omni `96bbf61`](https://github.com/tc-mb/llama.cpp-omni/tree/96bbf61710d44421ea54e63f5194f9b17c20d877),
  with reviewed local source patches in `scripts/prepare_ornith_les.py`.
  Routed GGUF extents retain their packed quantization and real top-eight
  routing. Dense derivatives are private; experts stay in the original file.
- The fast chat uses OS page cache; the comparison also has bounded own-cache
  and LES variants. Metal offloads supported graph operations; CPU reference
  controls remain separate. Target KV is F16; Flash Attention is disabled.
- Swiftlet/QPack is preserved as a separate experiment; it is not the engine
  serving the installed Ornith GGUF in fast chat or the LAN API. A trained
  phone model has no silent full-model CPU fallback in the native LES path.
- `OrnithLESChatGeneration` supplies canonical tokenizer/template, reasoning
  channel separation, generation budgets and prefix accounting. The new exact
  recurrent checkpoint is bounded to 128 MiB and retains attention KV in place.
  It is a conversation-boundary checkpoint, not arbitrary speculative rollback.
- `LocalModelAPIServer` provides authenticated `/v1/models` and
  `/v1/chat/completions`, SSE, cancellation and validated function calls. The
  computer executes tools. The process owns one shared chat/API model instance.
- iOS device, iOS Simulator build controls and macOS app are supported. Windows
  hosts the client/development tools, not this Apple runtime. The last
  independently confirmed private delivery is build 206; the new lifecycle
  and batch changes have a separate pending release gate. Build 215 stopped in
  a canonical-tokenizer regression test before the app/sign/archive/upload
  stages. The test assumed the HF template's behavior instead of the installed
  GGUF's actual behavior with thinking off; that assumption has been corrected.
  Passing local metadata controls does not establish a delivered phone update.
- Historical phone build 123 measured 4.299/4.082 decode token/s for initial/warm
  page cache on its fixed 128-token comparison. Later screenshots show 5.54
  token/s on a short run. Neither is a DFlash benchmark. The user's 32K/Ultra
  observations do not establish actual near-full 65K quality or memory.

## Checkpoint and compatibility findings

The pinned [official draft config](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash/blob/942d20fee84d04c9ecbf18e9c6fc4144914a1d5d/config.json)
and repository metadata specify **385,906,176 BF16 parameters**, a
**771,819,680-byte** Safetensors file, six draft layers, hidden size 2048,
vocabulary 248320 and block size 16. Five layers use a 4096 sliding window;
one uses full attention. This is DFlash v1, not DFlash2.
The block includes one anchor, so size 16 allows at most 15 draft proposals.
An exact HTTP-range read of **7328 header bytes**, with zero tensor-payload
bytes read, independently finds 69 BF16 tensors and `fc.weight` shape
`[2048,16384]`. There is no private token-embedding table or output head;
the draft must share these with the target. The full weight checksum has not
been verified because the tensor payload was not downloaded.

The eight target layer outputs are `[1,6,11,16,22,27,32,37]`. The inspected
[GGUF converter](https://github.com/ggml-org/llama.cpp/blob/d454a204ef2b63af357744809c05af897611a72e/conversion/qwen.py)
maps them to layer **inputs** `[2,7,12,17,23,28,33,38]`, copies the target
tokenizer and records draft sliding windows and M-RoPE metadata. Losing this
offset or treating all six draft layers as full attention is incorrect.
Both [target](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/10fbf86fed7ecee4a061f8b499a618f46001cac1/config.json)
and draft declare 262144 positional capacity. That is metadata, not proof of
usable context on an iPhone. Draft and target have different attention head
geometry; equal hidden size and correct feature wiring are the relevant bridge.

| Question | Source-based result |
| --- | --- |
| Official draft with installed Q4_K_M target? | Plausible after draft conversion and bridge work; not verified. Keep the existing target GGUF. Correct verification preserves the **quantized target's** distribution, not the original BF16 model's logits. Acceptance may change after quantization. |
| Does llama.cpp support DFlash? | Yes. Even our pinned omni source has `src/models/dflash.cpp`, `common/speculative.cpp`, target-feature APIs and recurrent rollback support. Our CMake explicitly disables `LLAMA_BUILD_COMMON`; the public `flo_decode` returns only last-position logits and sets `n_rs_seq=0`. DFlash is not exposed by the app. |
| Need vLLM/SGLang/Transformers? | No for a native port. They are alternative reference/deployment paths. Speculators supplies training, conversion and model definitions, not a Swift iOS runtime. |
| Reuse caches and memory management? | Target expert I/O/cache, model storage and budget admission can stay. Draft KV and speculative recurrent snapshots need additional explicit budgets. The existing boundary checkpoint alone is insufficient. |
| Run the draft separately? | A second native context can share target token embeddings/output weights. It still needs eight target feature streams and block verification. Two independent text-only OpenAI endpoints cannot implement this algorithm. |
| CUDA required? | Not by DFlash itself. z-lab has MLX Apple Silicon code; llama uses backend-independent graphs. Its Python Transformers CLI chooses CUDA and timings synchronize CUDA. Python MLX is not a drop-in iOS Swift component. CPU/hybrid drafting is possible engineering, with no demonstrated speed advantage here. |
| Extra files/dependencies? | Draft GGUF plus validated metadata/tokenizer provenance, dense draft graph/context, feature capture, all-position logits, accept/reject sampler, rollback, draft KV and cancellation ownership. Reuse native Metal/C++ rather than shipping Torch/Python. No need to download the large BF16 target or re-quantize its GGUF. |
| Thinking/stream/tools? | Compatible in principle because only verified tokens enter existing parsers. Must test thinking-budget closure, stop tokens and complete structured tool calls across block boundaries. Never stream rejected draft tokens. |

Sources for the runtime findings: pinned
[DFlash orchestration](https://github.com/tc-mb/llama.cpp-omni/blob/96bbf61710d44421ea54e63f5194f9b17c20d877/common/speculative.cpp),
[shared draft graph](https://github.com/tc-mb/llama.cpp-omni/blob/96bbf61710d44421ea54e63f5194f9b17c20d877/src/models/dflash.cpp),
[recurrent rollback](https://github.com/tc-mb/llama.cpp-omni/blob/96bbf61710d44421ea54e63f5194f9b17c20d877/src/llama-memory-recurrent.cpp),
[MLX generation](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py),
[Speculators converter](https://github.com/vllm-project/speculators/blob/94db8b8f9770cb6b8f0ff655029ebbe5900bb128/src/speculators/convert/dflash/converter.py).
The current upstream converter is not registered in the inspected pinned omni
conversion file; conversion must be pinned and its emitted metadata checked
against the pinned graph. Do not silently upgrade the whole inference engine.

The inspected Speculators converter is also not a drop-in conversion route for
this target's raw config: automatic layer selection reads the root
`num_hidden_layers`, while Ornith puts it in `text_config`. A source-level
metadata-method probe reproduces that missing-key error; supplying the eight
validated input-layer IDs avoids that branch. This probe mocks I/O and model
constructors; it is not a successful full conversion or inference. A native
prototype must adapt the config deliberately. Speculators' newer inspected
`4f9cbf8` changes navigation documentation only; its converter has the same
behavior.

Keep the installed target's tokenizer, chat template and stop policy authoritative.
The draft config's EOS is `248044`; the installed GGUF declares EOS `248046`.
Do not replace target stopping behavior with draft metadata. The HF config and
GGUF chat templates also differ, so merely copying a reference tokenizer config
does not establish chat/tool parity. Validate token IDs and special-token
handling against the installed GGUF before exposing a converted draft.

## Memory findings

These are payload calculations, **not measured process footprint**. F16 draft
KV costs `2 × 2 × 8 × 128 × (context + 5 × min(context,4096))` bytes.

| Context | BF16 draft file | Draft F16 KV payload | Sum before other additions |
| --- | ---: | ---: | ---: |
| 8192 | 771,819,680 | 117,440,512 | 889,260,192 |
| 32768 | 771,819,680 | 218,103,808 | 989,923,488 |
| 65536 | 771,819,680 | 352,321,536 | 1,124,141,216 |

The existing metadata test's conservative 65K process plan is 4,873,455,907
bytes at its stated baseline and physical batch four. Adding just the last row
gives 5,997,597,123 bytes, already above 5.5 GB. Actual available iOS memory,
allocator scratch, logits and recurrent rollback still need admission/measurement.
This estimate does not establish that every possible streaming draft design
fails. Draft-only quantization, a smaller adaptive block or file-backed draft
weights are separate experiments; retain the target's original quantization.

Compare two rollback designs in the prototype: native per-position recurrent
snapshots, or one bounded block-start snapshot followed by re-evaluation of the
accepted prefix after a rejection. The second design can reduce memory but adds
target work on rejection. It needs a separate slot from the conversation
checkpoint and a combined byte cap; it is not implemented by the current single
checkpoint ABI. Measure both instead of assuming per-position copies are free.

The inspected Python MLX implementation retains all eight target feature
streams during prefill when any draft layer is full attention. At 65K those
features alone occupy **2–4 GiB** for two/four-byte activations. A literal port
is unsuitable for our budget. Consume features incrementally: a four-position
F32 feature block is 262,144 bytes, excluding draft KV and recurrent state.

## Reproduction and admission gates

Run `python scripts/inspect_ornith_dflash.py --context 65536 --output audit.json`.
It downloads only pinned public JSON, verifies config digests and weight
identity from metadata, and records zero downloaded weight bytes. Add
`--inspect-tensor-header` for the bounded header-only range check. Seven controls
test geometry, layer offsets, memory arithmetic, invalid configurations and
honest unverified status. This is not a converter or a model benchmark.

Next isolated prototype gates, in order:

1. Verify converted draft metadata, shared head/embedding wiring and eight
   feature tensors against the reference; no target-weight rewrite.
2. Test target logits at **every** proposal position. Reject at the first,
   middle and final positions, restore attention **and DeltaNet/conv state**,
   and compare the continuation to fresh ordinary inference.
3. Require greedy token parity. For nonzero temperature, implement exact
   rejection/residual sampling against the same target sampler; same random
   seed alone does not imply the same token sequence. Test the distribution.
4. Test stop markers, tool JSON, thinking budget, cancellation, prefix reuse
   and foreground transitions. Only accepted tokens may reach SSE or history.
5. On the phone compare off/on in alternating order, at least 128 emitted
   tokens per comparable run: 8K first, then 32K/65K only after memory admission.
   Record draft/verify time, proposed/accepted tokens, I/O, GPU dispatches,
   TTFT, RAM and thermal state. Keep DFlash off if it slows down or fails a gate.

No extrapolated 10–15 token/s target is claimed. DFlash accelerates decode only
when accepted tokens amortize draft, verify and rollback work; it does not
remove long-prompt prefill or increase context capacity by itself.

## Related cache regression found during the audit

The ordinary installed-model tokenizer was also checked independently, without
inference. On a public fixture, the first generation prompt has 15 tokens with
Thinking off and 13 with it on. With the actual GGUF template the complete
prompt remains a prefix with Thinking off, but changes with Thinking on when
the answer is rendered as history without reasoning. The HF config template
instead changes it in both modes. Their SHA256 digests are respectively
`f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b`
and `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`.
The eight-token history boundary before the assistant's generation suffix
remains exact in all four cases. The new runner captures there and verifies
token-suffix equality; it does not edit the original chat template. Matching
Swift canonical-tokenizer and cache policy regressions were added. This
establishes a tokenizer-level invalidation mechanism, not an on-phone cache
speedup; the full app gate and physical warm/fresh test are still required.
