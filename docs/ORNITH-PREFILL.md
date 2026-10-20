# Ornith prompt batching and work profiles

Built and privately delivered as **0.1.0 (126)** on the isolated
`codex/ornith-prompt-controls` branch, runtime source `6c1084c87b35c749a5152c140c4828e9dde84126`.
[Complete delivery run 39250964590](https://github.com/Dhafen/flash-local/actions/runs/39250964590) passes integration, archive/signing and
fresh independent Apple availability. All 64 platform/appearance captures were
actually reviewed. The 32 selected native/chat/tokenizer Swift controls, 48 new
full-graph batch variants, 585 app tests, 483 package invocations (9 skips)
and nine actual API tests pass, alongside the retained reference/Swiftlet/QPack
gates. Actual-phone gains from larger batches remain unmeasured.

The existing GGUF, ordinary GGUF/REAM chats and build-125 base branch are retained.
No trained weights are downloaded or evaluated on the development/build hosts.

## Which batch size changes what?

[LM Studio describes evaluation batch size](https://lmstudio.ai/docs/developer/rest/load)
as the number of input tokens evaluated together. In this app, **evaluation**
selects the prompt chunk passed to the native wrapper: 1, 4, 8, 16 or 32 tokens.
The pinned engine's logical `n_batch` ceiling remains 32. **Physical batch**
selects the internal `n_ubatch` limit: Auto, 1, 4, 8 or 16. The wrapper splits
larger logical chunks into these microbatches and retains complete SSM/KV state.
The measured largest physical execution is exported separately from the limit.

The original default is unchanged: evaluation 4, physical Auto. Its existing
in-process gate selects 4-token or independently admitted serial processing.
Every nondefault configuration requires a separate short original-weight
F32/double reference check, exact top-eight expert sets, the unchanged absolute
0.01 / relative-L2 0.001 logit bounds, and a callback-free replay. The control
must actually execute the requested physical dimension. Model digest, native
revision, logical/physical configuration and 4096-token allocation are bound to
the proof. A failed check cannot activate that setting; select Auto to return.

Larger batches can improve prompt processing and time to the first answer.
They do not directly make single-sequence autoregressive decoding parallel.
More tokens can select more distinct experts and create larger GPU scratch
allocations, so a larger setting can also be slower or fail memory admission.
Desktop settings such as 512 are not exposed without an implementation and
validation for this streaming engine.

## Memory and unused outputs

The physical-batch staging allowance covers eight distinct experts per token:
32 slots for batch 4, 64 for batch 8 and 128 for batch 16. Fixed weights, KV,
SSM, process baseline and allocator/runtime reserve are budgeted before expert
staging. A larger graph receives an additional conservative scratch allowance
derived from recurrent-state, attention-score and vocabulary shapes. This is a
preallocation estimate, not a measured peak. The live iOS headroom and the
5.5 GB ceiling remain admission/cancellation limits.

The native wrapper only returns the last microbatch's logits. For preceding
microbatches in the same evaluation call, explicit zero output flags omit
unused final-output work. The pinned Qwen graph selects output rows after its
transformer layers. All actual experts, recurrent state and attention/KV work
must still match the serial full-graph control. Default four-token calls have
no intermediate microbatch and retain their previous output selection.
Counters record actual skipped intermediate output batches.

## Compare on the phone

Open **Ornith → Snabbchatt → Promptinläsning och batch**. A first candidate is
evaluation **32**, physical **8**. On iPhone the inline menus currently show
their values without separate visible labels: the **upper** menu is evaluation,
the **lower** is physical. Mac displays both labels. Explicit visible iOS labels
remain a small presentation follow-up; both Picker accessibility labels are in
the source. **Kontrollera och ladda om** applies a pending
choice only after admission; chat/API use is blocked while a changed setting
awaits reload. Stop the API before changing load settings.

**Jämför vald batch med Auto** runs a fixed long prompt in the order Auto,
candidate, candidate, Auto. Every run creates and releases its own native
context, uses unchanged greedy sampling/thinking, and allows 129 generated
tokens to cover at least 128 timed decode forwards. An earlier EOS is retained
and makes an undersized comparison fail. User system prompts/profiles do not
enter the benchmark.

The report includes input/output token IDs, actual batch dimensions, separate
prefill/decode timings and I/O, Metal dispatch/buffer/time counters, footprint,
Low Power Mode, thermal states, memory admissions and the numerical proof.
Exact complete output IDs and matching prompt IDs, sufficient length, admitted
memory, no Low Power Mode and comparable thermal states are needed to label
the four runs comparable. No engine is automatically selected. OS page-cache
coldness remains uncontrolled; ABBA ordering does not eliminate that limitation.
Repeat a promising result before claiming a sustained gain. Save/share the
report explicitly from **Spara batchmätning i Filer**; no report is uploaded.

## Host control boundary

[Fast native run 39250359772](https://github.com/Dhafen/flash-local/actions/runs/39250359772)
at source `78b34633f4125cdb5eae5e097f480d212cfa1a59` passes 16 variants for each
of three generated fixtures: F32, mixed Q4_K/Q6_K and 256 experts/top-eight.
All requested physical dimensions are actually executed, skipped intermediate
outputs are observed, complete routed expert sets agree, and compared full
logit arrays have zero maximum error against serial resident execution.
Callback-free execution and recurrent/KV reset replay also pass. These 48
variants contain no trained weights and establish no iPhone speed measurement.

The preceding [native/Swift run 39235302412](https://github.com/Dhafen/flash-local/actions/runs/39235302412)
at source `720a76346763965e2ecdef5d95d4028e45e1460a` passes the initial batch
policy, 16 LES Swift functions and isolated Mac/iPhone/Simulator builds. It
predates the additional scratch allowance, output skipping and UI/profile
integration. Those changes pass the complete [build-126 integration run](https://github.com/Dhafen/flash-local/actions/runs/39250964590).
These are host correctness/build controls; larger-batch trained-model admission
and sustained prefill/decode timings still require the physical phone.

## Useful controls for programming and security review

**Systemprompt och arbetsprofil** offers editable programming and security-review
starter instructions plus an empty/custom choice. Like [LM Studio presets](https://lmstudio.ai/docs/app/presets),
these make reusable instructions convenient. This initial implementation stores
one current editable system instruction; it is not an LM Studio preset importer
or a user-named preset library. Choosing a profile changes no sampler, model,
batch, quantization or routing setting.

The system prompt persists locally, is limited to 32 KiB and counts toward the
context. Changing it invalidates any incompatible exact-prefix reuse, so the
next question may require fresh prompt evaluation. The API snapshots it at
server start and uses it only when the client supplies no system message. An
explicit client system message takes precedence. Generated diagnostics retain
prompt token IDs, including system-token IDs, only on explicit local export.

The UI also shows prompt throughput, prefix reuse, actual batches and context
usage. Total context remains 4096 for prompt, reasoning and answer. Temperature,
top-p, seed, total output allowance and optional numeric thinking cap are kept
from build 125. Their neutral defaults are unchanged.

Structured tool calls/tool results and JSON-schema-constrained output remain
separate integration work for coding/security agents. A system instruction or
text-only OpenAI-compatible endpoint does not establish working tool execution.
The app does not execute tests/scans/tools or claim that a profile improves the
underlying model's benchmark quality. General trained-model quality and usable
8K/32K/65K context still need physical validation.
