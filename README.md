# Flash Local

An independent fork of [Minirun](https://github.com/nanguoyu/minirun-app), developing local REAM/MoE inference from an iPhone's internal flash.

Target: the 56.1 GB Qwen3.8-Flash-Next-REAM GGUF, a total process memory budget of approximately 5.5 GB, and installation through a private internal TestFlight build. Model weights are downloaded separately and never bundled with the app.

## Development status

Implemented foundations include bounded GGUF indexing, mixed GGML dequantization, row-streamed MLX projections, a bounded expert cache, PLE hashing, GDN/hyper-connection primitives, chunked FP16 text-attention KV storage, internal Documents/Models storage and descriptor-bound integrity verification.

The iPhone chat route now connects a fully verified local GGUF to the canonical llama.cpp tokenizer/Jinja bridge and streamed Qwen4Exp generation. It includes resumable bounded downloads, cancellation, chat persistence and live memory/I/O figures. Complete sessions match the pinned CPU reference on a tiny untrained fixture; this does not validate the 56.1 GB weights. The initial app policy is 1,024 context tokens and a 4 GB process budget.

New TestFlight delivery, full-model logit agreement, offline iPhone generation, measured phone memory/performance and 65K context remain unverified. Device Release compilation and automatic distribution export are under validation. The inherited model catalog is not evidence of REAM support.

See [implementation and validation](docs/REAM-IMPLEMENTATION.md) and the [draft implementation PR](https://github.com/Dhafen/flash-local/pull/1). GitHub Actions builds on Apple silicon with pinned Xcode 26.2. Actual model inference is not run in the cloud.

## Build

The app uses the Flash Local identity and its own bundle identifier. Original internal schemes and module names are retained. Signing and TestFlight distribution remain pending. On an Apple-silicon Mac with Xcode 26.2 selected, run `python3 scripts/build_tokenizer_xcframework.py` before resolving Swift packages or running xcodebuild. This builds the pinned vocabulary-only C runtime locally for Mac, iPhone and Simulator; the generated library is not committed. CI runs that preparation automatically. Run the xcodebuild gates in [AGENTS.md](AGENTS.md) and [the implementation notes](docs/REAM-IMPLEMENTATION.md). A Windows host can drive the hosted build.

## License and origin

Upstream source: Copyright (c) 2026 Dong Wang, Apache-2.0. Keep [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md). Minirun's name/logo are governed by [TRADEMARKS.md](TRADEMARKS.md). Vendored GGML code carries its full MIT notice. Model weights have their own license, including the original Qwen Community License; they are not covered by this repository's code license.

The [archived upstream README](docs/UPSTREAM-README.md) describes the original Minirun distribution. Its download/TestFlight links belong to that original product.
