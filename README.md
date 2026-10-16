# Flash Local

An independent fork of [Minirun](https://github.com/nanguoyu/minirun-app), developing local REAM/MoE inference from an iPhone's internal flash.

Target: the 56.1 GB Qwen3.8-Flash-Next-REAM GGUF, a total process memory budget of approximately 5.5 GB, and installation through a private internal TestFlight build. Model weights are downloaded separately and never bundled with the app.

## Development status

Implemented foundations include bounded GGUF indexing, mixed GGML dequantization, row-streamed MLX projections, a bounded expert cache, PLE hashing, GDN/hyper-connection primitives, chunked FP16 text-attention KV storage, internal Documents/Models storage and descriptor-bound integrity verification.

The iPhone chat route connects a fully verified local GGUF to the canonical llama.cpp tokenizer/Jinja bridge and streamed Qwen4Exp generation. It includes resumable bounded downloads, cancellation, chat persistence and live memory/I/O figures. Complete sessions match the pinned CPU reference on a tiny untrained fixture; this does not validate the 56.1 GB weights. The current policy is 1,024 context tokens and a 5.5 GB process ceiling.

Private TestFlight version **0.1.0 (63)**, app source `ddfa752`, is uploaded and Apple confirms `VALID`, `INTERNAL_ONLY` and `IN_BETA_TESTING` for the configured own-tester group. [Full delivery run 38877522212](https://github.com/Dhafen/flash-local/actions/runs/38877522212) succeeded, including all test/build gates, device archive, actual Apple signature/profile, both memory entitlements, privacy-manifest checks and private availability. See the [delivery record](docs/TESTFLIGHT.md).

The user reports text generation on the physical iPhone with build 56, with unusably slow performance under its old 4 GB policy. Build 63 adds packed Metal products, grouped experts, an admitted dense cache, unused prompt-head elision and saved partial measurements after Stop; see [performance work and validation](docs/REAM-PERFORMANCE.md). It uses the same model file: update without uninstalling or downloading the weights again. Its phone speed and peak memory remain unmeasured. Full-model logit agreement, offline proof, 65K context and comparative quality also remain unverified. Structured tool calling and the optional LAN model server are not yet implemented. The inherited model catalog is not evidence of REAM support.

The quality target remains Qwen3.8-27B-class capability. See [model candidates and the paired local evaluation contract](docs/REAM-QUALITY.md), [implementation and validation](docs/REAM-IMPLEMENTATION.md) and the [draft implementation PR](https://github.com/Dhafen/flash-local/pull/1). GitHub Actions builds on Apple silicon with pinned Xcode 26.2. Actual model inference is not run in the cloud.

## Build

The app uses the Flash Local identity and its own bundle identifier. Original internal schemes and module names are retained. On an Apple-silicon Mac with Xcode 26.2 selected, run `python3 scripts/build_tokenizer_xcframework.py` before resolving Swift packages or running xcodebuild. This builds the pinned C runtime for Mac, iPhone and Simulator: a vocabulary-only tokenizer path and a separate, bounded CPU numerical diagnostic. The generated library is not committed. CI runs that preparation automatically. Run the xcodebuild gates in [AGENTS.md](AGENTS.md) and [the implementation notes](docs/REAM-IMPLEMENTATION.md). See [private TestFlight delivery](docs/TESTFLIGHT.md) for signing, upload and availability checks. A Windows host can drive the hosted build.

## License and origin

Upstream source: Copyright (c) 2026 Dong Wang, Apache-2.0. Keep [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md). Minirun's name/logo are governed by [TRADEMARKS.md](TRADEMARKS.md). Vendored GGML code carries its full MIT notice. Model weights have their own license, including the original Qwen Community License; they are not covered by this repository's code license.

The [archived upstream README](docs/UPSTREAM-README.md) describes the original Minirun distribution. Its download/TestFlight links belong to that original product.
