# Ornith 1.5 35B-A3B pivot

The user selected Ornith as the new primary model after the REAM phone trial.
This changes the preferred artifact, while preserving the local-only execution
and 5.5 GB process ceiling. The original 50–70 GB preference is not a minimum:
the official Q4_K_M file is 21,713,463,040 bytes. REAM remains selectable and
its installed file and saved conversation are retained independently.

## Pinned inputs

- [Official source](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/10fbf86fed7ecee4a061f8b499a618f46001cac1).
- [Official GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/tree/12393612fd4f730ff5aadc23e9b8f9648aa49ceb):
  `Ornith-1.5-35B-Q4_K_M.gguf`, SHA-256
  `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
- Exact 10,990,848-byte header SHA-256
  `961df6ceaeedc2fbc0c84dfc16d8ba3ebf1d7e67506d10096a4d3a52d8a0836b`.
  `scripts/fetch_ornith_metadata.py` requires this hash and an exact HTTP 206
  range. No tensor bodies are fetched on the development host or CI.
- Embedded chat template SHA-256
  `f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b`.
  Canonical BPE/Jinja and EOG IDs still come from pinned llama.cpp.
- The publisher declares MIT licensing. Its pinned 1,073-byte notice is
  bundled in the existing offline notices resource and checked during archive
  inspection. No unconfirmed upstream base checkpoint is invented in reports.

## Architecture and implementation

The header declares `qwen35moe`, 41 stored blocks with one NextN/MTP block.
Only the 40-block main trunk executes: thirty GDN blocks and ten full-attention
blocks. Hidden width is 2,048; there are 256 routed experts with eight selected
per layer, expert/shared widths 512, sixteen QK and thirty-two value GDN heads,
state width 128, and two KV attention heads of width 256. The 753 tensors use
F32, Q4_K and Q6_K, all supported by the packed streaming kernels.

`Qwen35MoEDecoder` implements the ordinary residual/RMSNorm graph, gated
attention, shared/routed SwiGLU and **SiLU** GDN output gating. It does not reuse
REAM's hyper-connections, PLE or sigmoid GDN output gate. Text MRoPE uses the
artifact's unscaled base and dimensions; visual inputs, alternative recurrent
layouts and RoPE scaling are refused. MTP weights are verified with the whole
file but not executed. There is no speculative/draft-model path.

RMS rows use the pinned CPU operation's Float32 squares, Float64 sum and
Float32 mean/scale/products, bounded to 65,536 elements. This explicitly pays
for a small CPU/GPU transfer to avoid a GPU reduction/rsqrt ULP difference
being amplified by activation quantization or FP16 KV rounding. Tests compare
these rows bit for bit against CPU checkpoint inputs and record a transfer-
inclusive synthetic benchmark. Projection weights remain streamed; this does
not introduce a CPU whole-model inference path. Full graph gates still apply.

The mixed fixture exposed one V projection just across a half rounding boundary
(`0.23089598` CPU versus `0.23089603` packed Metal). Its single differing cached
FP16 value then affected subsequent positions. Small K/V projections therefore
use the unchanged pinned CPU `vec_dot` with its canonical operand conversion.
Each is limited to one verified packed tile, at most 4 MiB, 1,024 output rows
and 65,536 input columns. The same protected cache and reader own those bytes;
there is no file or model opener in this helper. Expert projections remain on
Metal. Reports identify `canonicalCPUKVCalls` and count explicit CPU handoff
copies even on a resident hit. Tests compare raw results bit for bit against
the CPU graph and record paired transfer/I/O-inclusive host timings. This is
a numerical stability tradeoff pending actual phone performance measurements.

`GGUFStreamedDecoder` dispatches explicitly by architecture; the existing chat,
prefix checkpoints, cancellation, packed projections and bounded I/O are shared.
The first Ornith app policy is 4,096 context tokens, up to 512 new tokens,
3.5 GB packed payload capacity, 64 MiB raw slots and the existing 1 GB working
reserve, all admitted inside the 5.5 GB ceiling. This is a configured limit,
not proof that 4K or 65K has been physically tested. The memory arithmetic for
65,536 positions is 1,342,177,280 KV bytes plus 62,914,560 recurrent-matrix bytes;
that arithmetic is not measured RAM and requires a different cache allowance.

Whole eligible dense matrices receive protected admission first. The unused
part of the same 3.5 GB allowance admits selected expert tiles through an
identity-linked O(1) LRU. Expert churn cannot evict protected dense entries.
The original REAM policy leaves this LRU disabled. Reports expose the
evictable capacity, payload, hits and evictions; copied pending groups keep
their existing 32 MiB bound. Synthetic tests check byte-exact reuse, LRU order,
poisoned released slots, independent products, zero warm reader bytes and
full-graph agreement with/without this cache. Host timings do not establish
phone throughput or cache hit rates.

The app defaults to Ornith and has a model picker. Switching releases the old
engine first, then restores that model's separate saved chat. It does not
remove an installed artifact or resume an unrelated download. Empty interrupted
turns are omitted as pairs from model context, fixing the reported consecutive
user-role failure while keeping visible failed attempts.

## Verification status

TestFlight **0.1.0 (80)** at `c17b9cdc6aefd6c5471b69c1c321e7bbe00807a4`
passed the full delivery in
[run 38926302018](https://github.com/Dhafen/flash-local/actions/runs/38926302018).
The numerical gates cover
five-position FP32 and mixed F32/Q4_K/Q6_K synthetic graphs, complete canonical
chat sessions, all-state rewind/cancellation, app integration and local CPU
comparison. The synthetic fixtures include an unused MTP block and non-uniform
norm scales. The mixed comparison's worst maximum absolute error was
`5.9604645e-8` and worst relative L2 was `1.6299255456418377e-7`; all compared
FP16 value-cache entries matched. Thresholds remain 0.01 and 0.001, respectively.
Reference logits and fixtures were not relaxed. No trained-model inference
runs on hosted hardware.

The same run passed 569 app tests and 474 package tests (nine explicit
environment skips), both iOS compilation stages, all twelve light/dark Mac/iOS
layout captures and actual Apple signing with both memory entitlements.
The captures were also inspected visually. Upload exited zero; Apple confirmed
`VALID`, `INTERNAL_ONLY` and `IN_BETA_TESTING`. A separate read-only audit
confirmed the exact existing private group and unchanged sole-owner membership.
The device app contains no model weights. See the [delivery record](TESTFLIGHT.md).

Earlier candidate 77 passed its test/build/signing gates but failed on an
Apple GET timeout before upload; it was never delivered. The client now
retries bounded read failures, including a timeout during TLS response reading,
while uncertain writes never repeat. Thirty-two local Python tests pass.

No Ornith phone speed, peak RAM, offline proof, full trained-model numerical
agreement or quality against Qwen3.8-27B is established yet. Publisher coding
and agent benchmarks use different harnesses, thinking and context policies
from this initial greedy non-thinking app; those scores cannot be assigned to
the phone app. Structured tools/thinking history and the authenticated LAN
server remain future work after the text runtime passes physical validation.

## First physical measurement

Update the existing app through TestFlight without uninstalling. Select Ornith
in the information menu if it is not already selected. Its pinned 21.7 GB file
downloads separately over Wi-Fi into the app's internal Models directory;
existing REAM files and conversations remain separate. File verification must
finish before generation is enabled.

Start with a new Ornith conversation and two short questions, keeping the app
in the foreground between them. Save each completed run through the explicit
measurement action. If speed is unusable, Stop after a few output tokens and
save that partial measurement instead of waiting for the 512-token limit.
Compare `prefillSeconds`, `reusedPromptTokens`, `evaluatedPromptTokens`,
`decodeForwardPasses`, `decodeForwardSeconds`, `decodeIOBytesRead`,
`maximumObservedFootprintBytes` and the separate prefill/decode projection
statistics. The cache report includes protected/expert residency and eviction
counts; `canonicalCPUKVCalls` separates the bounded CPU projection work. Reader
bytes are returned application reads, not a physical flash/controller counter.

After a normal short run succeeds, repeat with airplane mode and Wi-Fi disabled
to establish offline execution. The JSON's default `airplaneModeVerified=false`
is not automatically changed by that manual test; record the actual phone
conditions separately. The information menu's short numerical diagnostic
compares local CPU and streamed logits for the same verified artifact, prompt
IDs and positions. Passing those positions does not certify every context or
task. Longer-context and paired capability tests follow only after this basic
run is usable; there are no phone results to publish yet.
