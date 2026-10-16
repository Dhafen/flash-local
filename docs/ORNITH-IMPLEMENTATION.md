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

`GGUFStreamedDecoder` dispatches explicitly by architecture; the existing chat,
prefix checkpoints, cancellation, packed projections and bounded I/O are shared.
The first Ornith app policy is 4,096 context tokens, up to 512 new tokens,
3.5 GB packed payload capacity, 64 MiB raw slots and the existing 1 GB working
reserve, all admitted inside the 5.5 GB ceiling. This is a configured limit,
not proof that 4K or 65K has been physically tested. The memory arithmetic for
65,536 positions is 1,342,177,280 KV bytes plus 62,914,560 recurrent-matrix bytes;
that arithmetic is not measured RAM and requires a different cache allowance.

The app defaults to Ornith and has a model picker. Switching releases the old
engine first, then restores that model's separate saved chat. It does not
remove an installed artifact or resume an unrelated download. Empty interrupted
turns are omitted as pairs from model context, fixing the reported consecutive
user-role failure while keeping visible failed attempts.

## Verification status

Implementation and test fixtures are committed in stages. CI is validating
the new graph before a new TestFlight delivery; build 68 still contains REAM.
Five-position FP32 and mixed F32/Q4_K/Q6_K synthetic graphs, complete canonical
chat sessions, all-state rewind/cancellation, app integration and local CPU
comparison are required gates. The synthetic fixtures include an unused MTP
block and non-uniform norm scales. Thresholds remain 0.01 maximum logit error
and 0.001 relative L2. No trained-model inference runs on hosted hardware.

No Ornith phone speed, peak RAM, offline proof, full trained-model numerical
agreement or quality against Qwen3.8-27B is established yet. Publisher coding
and agent benchmarks use different harnesses, thinking and context policies
from this initial greedy non-thinking app; those scores cannot be assigned to
the phone app. Structured tools/thinking history and the authenticated LAN
server remain future work after the text runtime passes physical validation.
