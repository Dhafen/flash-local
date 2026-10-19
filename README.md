# Flash Local

An independent fork of [Minirun](https://github.com/nanguoyu/minirun-app), developing local MoE inference from an iPhone's internal flash.

Current target: the official 21.7 GB Ornith 1.5 35B-A3B Q4_K_M GGUF, a total process memory budget of approximately 5.5 GB, and installation through private internal TestFlight. The user selected this pivot after the 56.1 GB REAM trial. Model weights download separately and never enter the app bundle; installed REAM weights remain available as a separate model choice.

## Development status

The separate `codex/ornith-prompt-controls` branch adds
[validated prompt batch settings, an ABBA comparison and editable work instructions](docs/ORNITH-PREFILL.md).
Physical batches 1/4/8/16 and logical chunks through 32 pass 48 generated
full-graph controls. App integration/delivery and real-phone performance for
these new settings are still pending; the delivered checkpoint below is retained.

The isolated branch now implements an authenticated foreground
[local OpenAI-compatible API and numeric generation controls](docs/ORNITH-LOCAL-API.md),
including an optional thinking budget. Source-identical host controls exercise
real loopback TCP and compile the server for iOS. Integrated app checks,
private delivery and independent Apple availability pass. Physical-phone LAN,
adapter/USB reachability and explicit-budget answer quality still need testing.

**Private TestFlight 0.1.0 (125) is available**, app source `fdb7801742a3de0015e3047a055f555db41d9b98`.
The new **Ornith snabbchatt** is an isolated experimental page-cache chat using
the installed 21.7 GB Q4_K_M GGUF. It requires a fresh in-process numerical
admission, preserves real top-eight routing, streams reasoning separately,
reuses only exact token prefixes and exports measurements locally. The ordinary
GGUF/REAM engines and saved conversations remain available. Update without
uninstalling or downloading the model again. [Delivery and instructions](docs/TESTFLIGHT.md).

The supplied physical iPhone18,2 **build-123** comparison completed eight
128-token runs on one fixed prompt. Decode rates, initially empty/warm app cache:

| Engine | Decode token/s, first / warm |
| --- | ---: |
| Existing GGUF | 1.156 / 1.151 |
| Native expert cache | 3.675 / 3.664 |
| Native LES depth 4 | 4.044 / 3.834 |
| Native OS page cache | 4.299 / 4.082 |

All six native outputs are identical; page cache wins their comparison in both
runs. The ordinary engine has a different output digest, so an output-equivalent
four-way speedup is not established. Exact expert sets and the short independent
original-weight F32/double logit gate pass. Batch-four now also passes on the
trained phone model. Page-cache prefill reaches 7.253 / 7.189 prompt token/s;
decode is lower than in the [build-121 session](docs/ORNITH-PHONE-121.md).
The original quantized-operand CPU discrepancy remains visible.
OS-cache coldness is uncontrolled and process footprint does not represent all
file-cache RAM. [Measurements and limits](docs/ORNITH-PHONE-123.md).

Build 123 corrects a reproduced Q4_K Metal scale-precision error without
requantization. Six actual untrained GPU row controls now match CPU decoding
exactly; the hosted virtual GPU cannot run the complete native projection probe.
The phone report admits corrected batch-four execution; a separate quick-chat
screenshot shows 4.50 token/s and completed reasoning/answer UI.
This is not a 10–15 token/s claim or general quality validation.

The [complete delivery run](https://github.com/Dhafen/flash-local/actions/runs/39221402989) passes 26 selected native/chat/tokenizer Swift functions,
three independent synthetic F32 graphs, 63 routed and 18 callback-free variants,
three isolated Apple framework slices, the retained Swiftlet/QPack suites,
583 app tests, 483 package invocations (9 environment skips), nine actual API
controls and 56 actually reviewed captures. Device archive, signing, local-network
purpose text and independent private TestFlight availability pass. These host
controls do not establish physical throughput or phone API reachability.

Context remains 4,096 tokens for Ornith and 1,024 for the separately retained
REAM model, with a 5.5 GB process ceiling. General trained-model quality, 8K/32K/65K
operation, phone LAN/USB reachability and structured agent-tool contracts remain
unverified. The foreground text API is implemented and built.
Model weights are external artifacts and never enter the IPA. See the
[engine details](docs/ORNITH-LES.md), [Ornith implementation](docs/ORNITH-IMPLEMENTATION.md),
[separate Swiftlet/QPack experiment](docs/ORNITH-PERFORMANCE.md) and
[Bonsai feasibility review](docs/BONSAI-EVALUATION.md).

## Build

The app uses the Flash Local identity and its own bundle identifier. Original internal schemes and module names are retained. On an Apple-silicon Mac with Xcode 26.2 selected, run `python3 scripts/build_tokenizer_xcframework.py` before resolving Swift packages or running xcodebuild. This builds the pinned C runtime for Mac, iPhone and Simulator: a vocabulary-only tokenizer path and a separate, bounded CPU numerical diagnostic. The generated library is not committed. CI runs that preparation automatically. Run the xcodebuild gates in [AGENTS.md](AGENTS.md) and [the implementation notes](docs/REAM-IMPLEMENTATION.md). See [private TestFlight delivery](docs/TESTFLIGHT.md) for signing, upload and availability checks. A Windows host can drive the hosted build.

## License and origin

Upstream source: Copyright (c) 2026 Dong Wang, Apache-2.0. Keep [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md). Minirun's name/logo are governed by [TRADEMARKS.md](TRADEMARKS.md). Vendored GGML code carries its full MIT notice. Model weights have their own license, including the original Qwen Community License; they are not covered by this repository's code license.

The [archived upstream README](docs/UPSTREAM-README.md) describes the original Minirun distribution. Its download/TestFlight links belong to that original product.
