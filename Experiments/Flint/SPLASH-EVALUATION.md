# Splash and Flint on iPhone

This is a source review, not an iPhone benchmark or an integrated backend.
Splash source was inspected at `dbc46102317320a058809a7366b6c87f82725818`:
[Flint target](https://github.com/incoai/splash/blob/dbc46102317320a058809a7366b6c87f82725818/runtime/model/FlintTarget.cpp),
[draft](https://github.com/incoai/splash/blob/dbc46102317320a058809a7366b6c87f82725818/runtime/model/DFlashDraft.cpp),
[expert store](https://github.com/incoai/splash/blob/dbc46102317320a058809a7366b6c87f82725818/runtime/model/ExpertStore.cpp),
[capabilities](https://github.com/incoai/splash/blob/dbc46102317320a058809a7366b6c87f82725818/runtime/metal/DeviceCapabilities.cpp).

The current Flash Local Flint engine streams experts and uses Metal, but does
not include Splash, lookahead expert prefetch, or Flint DFlash2. Splash uses
model-shaped kernels, asynchronous expert reads and speculative verification.
Its reported results are from Macs and do not establish iPhone performance.

Splash's Flint draft consumes target activations after layers 1, 12, 24, 35
and 47 and proposes a block containing an anchor and up to seven tokens. Our
native interface currently exposes final logits, KV reset and route diagnostics,
not these intermediate activations, per-position verifier logits or partial KV
rollback. The published [draft GGUF](https://huggingface.co/incoai/Flint-DFlash2-GGUF)
includes a roughly 240 MB Q4_K_M file, but simply downloading it cannot add
speculative decoding to the existing engine.

For a future isolated experiment, first capture the target activations and
verify blockwise logits and KV rollback against the existing one-token path.
Then compare greedy tokens and, for sampling, use target/draft probabilities
and residual sampling rather than treating token equality as sufficient.
Separately measure prefetch predictions, actual top-6 routes, I/O, memory and
throughput. Any prefetch prediction must affect I/O only, never routing math.

The upstream runtime has macOS-specific capability checks and IOKit code; a
native iOS port requires verifying the phone's Metal capabilities and memory
budget. The existing SwiftUI and LAN server can remain the product interface.
No iPhone speedup, long-context result or model-quality equivalence is yet
verified for Splash or the draft.
