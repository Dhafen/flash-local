# Flint on iPhone: verified streaming, 65K context and Splash integration gates

This branch is **diagnostic and experimental**. No claim of 65K running
successfully, native iPhone throughput or trained-model quality is made here.
Do not merge into the shipped app or publish a new TestFlight build until its
physical-device gates pass.

## Intended execution architecture

- Retain the original 25,992,966,976-byte `flint-int4-q4_1.gguf` in the
  iPhone app's internal persistent storage, verified by its pinned SHA-256.
  The model must **not** need a 26 GB resident Metal allocation, an external
  SSD, a converted second copy or a cloud inference backend.
- Keep fixed tensors and embedding row access backed by their verified source
  where supported; stream only the six **actual** routed experts selected at
  each of 47 MoE layers into bounded shared Metal/CPU staging banks.
- Reuse exactly matching prompt-prefix KV states; never reuse shortened or
  changed prompt histories against truncated sliding-window caches.
- Change caching, I/O overlap, speculative **verification** or layout only.
  Do not alter expert selection, published Q4_1 weights, normalization, RoPE,
  attention sinks, window coverage or output-sampling probabilities.

## 65K hard boundary: current memory arithmetic

With the checked-in profile and the current production `F16` K/V policy:

| Item | Conservative bytes |
| --- | ---: |
| Non-routed tensors (including file-backed embedding upper bound) | 2,331,622,528 |
| Padded 65,536-token K/V | 1,934,098,432 |
| 24 expert staging slots, physical batch 4 | 47,185,920 |
| App/base reserve | 400,000,000 |
| Scratch reserve | 1,000,000,000 |
| **Estimated process budget** | **5,712,906,880** |
| Current hard ceiling | 5,500,000,000 |
| **Deficit** | **212,906,880** |

This is a *conservative estimate*, not measured residency. The current code
correctly **rejects** a 65K allocation. Raising limits, pretending that mmap
pages are always nonresident, changing the KV dtype or bypassing OS headroom
without measurement does not count as supporting 65K. The UI must not advertise
65K as usable before at least one physical-iPhone acceptance report proves
admission, sustained forward passes, and correct long-context retrieval.

To close the deficit without degrading accuracy, first measure the actual
embedding/output-head residency and GPU scratch high-water on the device,
then evaluate safe release of genuinely unused mapped pages and bounded
temporary buffers. Recalculate the upper bound from observed allocations,
not assumed savings. Only *then* offer a dynamic 65K selection gated by live
`ProcessFootprint`, with fail-closed cancellation when headroom vanishes.
If a phone cannot meet this, report a lower tested limit rather than silently
truncating its context.

## CPU–Metal acceptance before performance work

Build 250's attached iPhone JSON fails the numerical gate: 128/556 top-6
expert sets differ from ordinary CPU, and the Q4/F32 diagnostic still differs
in 2/556. Position 6 changes the best token in the ordinary Metal candidate.
The current fail-closed gate **must remain enabled**.

This branch adds a fourth, *diagnostic-only*, comparison of CPU Q4_1/F32
against Metal using matched F32 K/V, Flash Attention disabled and physical
batch 1. The resulting `matchedMetalQ4F32Control` field isolates whether
a remaining discrepancy comes from attention/KV/batching rather than CPU
Q8_1 activation rounding. It does not grant a serving admission.

After collecting the new report, identify the first different **intermediate
tensor** in the earliest diverging layer and correct the specific CPU/Metal
operator. Require an independent trained-model, per-layer route/logit
regression on-device before accepting a change.

## Splash is a design input, not an installed iOS backend

[Splash upstream](https://github.com/incoai/splash) 1.4.1 supports Flint
on **macOS** with DFlash2 speculative decoding, asynchronous expert access
and specialized Metal kernels. Its Flint release notes report a Mac benchmark,
not an iPhone result. Its GGUF serving example uses an Unsloth `UD-Q4_K_XL`
variant, **not** this app's verified original Q4_1 file. The current
[Flint draft](https://huggingface.co/incoai/Flint-DFlash2-GGUF) does not by
itself implement speculative verification.

A safe iOS implementation requires: verified intermediate activation taps,
draft/target tokenizer parity, multi-token target verification, correct KV
commit/rollback including SWA and full-attention layers, unbiased rejection
sampling for non-greedy outputs, and explicit memory/disk admission for the
draft. Predictions may prefetch **bytes** but may never substitute predicted
expert IDs for model routing. Benchmark normal target decode **before**
enabling speculation so a failed speedup never changes model quality.

For more ideas, Splash 1.4.1 describes releasing expired SWA K/V and INT8 KV.
Changing K/V precision **does** change numerical outputs: it is a separate,
opt-in quality-gated experiment, not a drop-in memory fix.

## Measurements and release gates

1. Real iPhone numerical gate: four positions, all computed top-6 routes,
   full logits, serving callback-free path; save the complete report.
2. Repeated full-model chat/tool sessions with identical inputs and sampling,
   compare token IDs and tool-call JSON; never accept a routing bypass.
3. Peak actual footprint and OS headroom at 8K/32K/65K, cold/warm,
   including long continuous 65K prompt plus **output tokens**.
4. Compare prefill, time-to-first-token, sustained 128+ token decode,
   bytes explicitly read, per-layer I/O wait, thermal state and power
   across physical microbatch and bounded expert cache settings.
5. If a Splash-like draft is ported, compare target-only output against
   speculative output with greedy *and* sampled exactness checks. Report
   accepted tokens/cycle and **actual** iPhone throughput, not Mac results.

## Already fixed independently in this branch

Flint's LAN request handler previously rejected bodies over 256 KiB even
when a model could tokenize much longer prompts. It now admits a bounded
2 MiB body for Flint only; the original 256 KiB Ornith limit remains.
This fixes an API transport bottleneck, **not** a context-memory or model
numerics problem. Long-context requests must still pass context and memory
admission before generating any output.
