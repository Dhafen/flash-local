# Isolated Flash Attention and DFlash phone tests

Implementation status: code added; native/app builds, synthetic controls and
delivery must pass before the user can run this test. **No trained DFlash or
phone Flash Attention result is claimed.** The ordinary chat, LAN server,
installed target GGUF, QPack/LES modes and their admissions remain unchanged.

## Test from the app

Open Ornith fast chat → **Prova Flash Attention och DFlash**.

1. Start with the **Flash Attention** comparison. It needs only the installed
   target. The separate context requests the fused attention graph, and records
   the runtime's actual graph flag. Aggregate GPU timing does not prove that
   every attention operator ran on Metal.
2. For **DFlash**, press **Hämta DFlash-draft · 783 MB**. The resumable bounded
   range downloader writes directly to private storage and verifies full SHA256.
   No second full file is created. The target is neither fetched nor rewritten.
3. Run **Kör kontroll och jämförelse**. Keep the phone foreground. Stop cancels
   inference through the existing native memory/cancellation governor.
4. Share the saved acceleration JSON. Load the ordinary model again after the
   experiment; chat history is retained. The experiment never promotes a backend.

Use 8K first. DFlash offers 8K/16K/32K, with a fresh memory admission for each
context and mode. At 32K its additional reserve is **1,068,032,352 bytes**:
782,819,680 actual GGUF file bytes + 218,103,808 F16 KV payload + 64 MiB of
draft scheduler/features/logit allowance. The existing 128 MiB recurrent
checkpoint reserve is reused by this benchmark; it does not retain a separate
conversation checkpoint. A live DFlash chat integration would need separate
checkpoint ownership/budget and remains follow-up work.

This is an allocation-capacity test with a short fixed public prompt. It does
not establish usable near-full 32K context, long reasoning quality or 65K.

## Model provenance

The optional file is the community publisher's
[BF16 GGUF conversion](https://huggingface.co/na0x2c6/Ornith-1.5-35B-A3B-DFlash-GGUF/tree/3427ea426d150d03c633b0d25b8a0626453b3f2f)
of [Ornith's official DFlash v1 checkpoint](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash/tree/942d20fee84d04c9ecbf18e9c6fc4144914a1d5d).
It is pinned by file length and SHA256
`abb048837823d068b71cea4d162076e5e2be62f504dc79bf08ac80d78c8bb732`.
This is not an official Ornith GGUF artifact or a separately authenticated
conversion lineage. A matching checksum proves identity to the pinned artifact,
not its numerical equivalence to the source Safetensors.

The conversion has 69 tensors, BF16 matrices and F32 norms, no embedding/head,
eight input-layer IDs `[2,7,12,17,23,28,33,38]`, five 4096 sliding layers and one
full attention layer. The native loader checks draft/target dimensions, mask
ID, every vocabulary token string, absent duplicate embedding/head and these
layer offsets. It shares target embedding/head buffers. The installed target's
template and stop policy are authoritative.
The inspected metadata contains exactly the same 248,320 token strings as the
installed target (length-delimited vocabulary digest
`b7f4906b5bf6a845baf3f41fdcdcd70f0f2f234eb702aabb830ce1536604e5d6`).
That is tokenizer metadata evidence, not draft/target inference validation.

## Correctness and measurement contract

- Ordinary engine → candidate → candidate → ordinary engine (ABBA), fresh
  contexts and exactly the same public prompt and greedy target sampler.
- Before each timed run, four full-vocabulary positions are compared with the
  ordinary engine, in observed and callback-free execution. The observed trace
  must retain the exact per-position/layer top-eight expert sets. Existing
  independent original-weight admission is also required.
- Logit limits remain max absolute 0.01 and relative L2 0.001, with identical
  greedy choices. These short controls do not establish full model numerics.
- At least 128 emitted tokens and identical output IDs in all four runs are
  required for a comparable speed report. EOS is respected; an early stop is
  recorded as insufficient length, never padded or called a successful result.
- DFlash proposes at most three tokens beside the target-chosen anchor. Target
  logits verify every proposal position. On rejection the complete recurrent/
  conv state and attention suffix are restored, then only the accepted prefix
  is re-evaluated. Rejected tokens never reach the report's emitted output,
  saved conversation or SSE. This first experiment is greedy only.
- The target's `n_batch` is bounded to 16 only in the draft experiment because
  extracted-layer buffers reserve that many rows. Features are consumed each
  microbatch, rather than retained for the whole prompt.
- Record actual prefill/decode/TTFT, process peak, memory plan, thermal state,
  low-power state, target expert I/O, Metal dispatches/command buffers/GPU time,
  proposed/accepted tokens, replay count and actual target tokens evaluated.
  Target I/O counters exclude demand-paged draft reads; no physical NAND-byte
  or OS-cold-cache claim is made. Thermal state is not temperature in degrees.

Generated untrained native controls cover all four accepted-prefix lengths,
recurrent/attention rollback and continuation against ordinary logits, plus a
CPU fused-attention graph comparison. They do not run a trained draft or prove
that a physical phone supports the same kernels. The actual paired-model test
is performed on the user's phone after a passing build.

Thinking budgets, nonzero-temperature rejection sampling, streaming tool calls
and foreground recovery remain in the ordinary backend. Extending DFlash to
them requires additional validation; this benchmark is not a replacement server.
