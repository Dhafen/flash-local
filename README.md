# Flash Local

An independent fork of [Minirun](https://github.com/nanguoyu/minirun-app), developing local MoE inference from an iPhone's internal flash.

Current target: the official 21.7 GB Ornith 1.5 35B-A3B Q4_K_M GGUF, a total process memory budget of approximately 5.5 GB, and installation through private internal TestFlight. The user selected this pivot after the 56.1 GB REAM trial. Model weights download separately and never enter the app bundle; installed REAM weights remain available as a separate model choice.

## Development status

**Private TestFlight 0.1.0 (113) is available**, with streamed Ornith thinking,
VM-backed LES page buffers and explicit phase-boundary ownership using the
installed GGUF. [Delivery and limits](docs/TESTFLIGHT.md).
Host synthetic/reference gates pass. Phone build 112 admits the first native
check at 114 MB, then stops before the next one with a 2.216 GB live footprint.
The isolated VM-buffer/lifecycle correction passes host/reference controls
and is delivered in build 113; its phone confirmation is pending. No LES speed
result has completed on the phone. The latest ordinary GGUF chat export from
build 113 measures 1.397 decode forwards/s over 110 forwards with thinking
enabled and 3.912 GB peak; three supplied exports contain that same generation.
It is shorter than the planned 128-token comparison and is not a controlled
backend benchmark. No physical
QPack or LES throughput has been verified. Existing GGUF/REAM paths remain
available and the LES experiment requires no model redownload.

The post-113 native routing observer now honours top-k row strides and compares
canonical traces across prefill/decode order. Its separate
[native-only validation](https://github.com/Dhafen/flash-local/actions/runs/39111809758)
passes 63 synthetic prefix/mode variants, all 18 selected Swift functions and
three Apple framework slices. These observer/admission changes are not in
installed build 113 and do not establish a phone speed improvement.

The [Ornith adapter and model switch](docs/ORNITH-IMPLEMENTATION.md) are delivered in **TestFlight 0.1.0 (84)**. They implement the actual `qwen35moe` trunk, including SiLU GDN gating, ordinary residual/norm blocks, shared/routed experts, pinned tokenization and correct exclusion of the stored MTP block. Small K/V projections use bounded pinned CPU kernels to preserve reference rounding; other projections and experts stay on Metal. Ornith is the default, with separate persisted chats and a model picker retaining REAM. Build 84 pairs eligible gate/up projections without changing their per-row arithmetic; phone performance for that optimization is pending.

Implemented foundations include bounded GGUF indexing, mixed GGML dequantization, row-streamed MLX projections, a bounded expert cache, PLE hashing, GDN/hyper-connection primitives, chunked FP16 text-attention KV storage, internal Documents/Models storage and descriptor-bound integrity verification.

The iPhone chat route connects a fully verified local GGUF to the canonical llama.cpp tokenizer/Jinja bridge and explicit Qwen35MoE or Qwen4Exp streamed decoding. It includes resumable bounded downloads, cancellation, separate persisted chats and live memory/I/O figures. Complete sessions match the pinned CPU reference on tiny untrained fixtures; this does not validate either trained model. The initial context policies are 4,096 tokens for Ornith and 1,024 for REAM, within the 5.5 GB configured process ceiling.

Private TestFlight version **0.1.0 (84)**, app source `b093bec6b3d4061b6693e07fdb3e4a6ab9a95606`, is uploaded and Apple confirms `VALID`, `INTERNAL_ONLY` and `IN_BETA_TESTING` for the configured own-tester group. [Full delivery run 38946605484](https://github.com/Dhafen/flash-local/actions/runs/38946605484) succeeded, including reference/app/package gates, all twelve reviewed Mac/iOS light/dark captures, the device archive, actual Apple signature/profile, both memory entitlements and privacy-manifest checks. A fresh read-only audit confirms the exact private group and unchanged sole-owner membership. See the [delivery record](docs/TESTFLIGHT.md).

The first user-provided Ornith phone report from build 80 measures **1.493 decode forwards/s** over 16 forwards, **16.105 s for 14 prompt tokens** and **3.872 GB sampled peak footprint**. It is a short first-turn measurement with no prompt reuse; it does not verify longer context, offline execution or numerical/quality agreement. The previous REAM build 63 measured 0.211 decode forwards/s and 4.633 GB peak; different models and prompt histories prevent isolating a runtime-only improvement. See [Ornith measurement and current optimization](docs/ORNITH-IMPLEMENTATION.md#phone-performance-and-paired-projections) and the historical [REAM performance](docs/REAM-PERFORMANCE.md).

Ornith uses bounded parallel reads, whole selected-expert transfers, packed GPU evaluation, foreground prefix retention and an expert LRU inside its 3.5 GB packed-weight allowance. Update without uninstalling; the installed model file is retained. Trained-model logit agreement, offline proof, 65K context and comparative quality remain unverified. Structured tool calling and the optional LAN model server are not yet implemented. The inherited catalog is not evidence that another architecture executes.

The quality target remains Qwen3.8-27B-class capability. See [model candidates and the paired local evaluation contract](docs/REAM-QUALITY.md), [implementation and validation](docs/REAM-IMPLEMENTATION.md) and the [draft implementation PR](https://github.com/Dhafen/flash-local/pull/1). GitHub Actions builds on Apple silicon with pinned Xcode 26.2. Actual model inference is not run in the cloud.

## Build

The app uses the Flash Local identity and its own bundle identifier. Original internal schemes and module names are retained. On an Apple-silicon Mac with Xcode 26.2 selected, run `python3 scripts/build_tokenizer_xcframework.py` before resolving Swift packages or running xcodebuild. This builds the pinned C runtime for Mac, iPhone and Simulator: a vocabulary-only tokenizer path and a separate, bounded CPU numerical diagnostic. The generated library is not committed. CI runs that preparation automatically. Run the xcodebuild gates in [AGENTS.md](AGENTS.md) and [the implementation notes](docs/REAM-IMPLEMENTATION.md). See [private TestFlight delivery](docs/TESTFLIGHT.md) for signing, upload and availability checks. A Windows host can drive the hosted build.

## License and origin

Upstream source: Copyright (c) 2026 Dong Wang, Apache-2.0. Keep [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md). Minirun's name/logo are governed by [TRADEMARKS.md](TRADEMARKS.md). Vendored GGML code carries its full MIT notice. Model weights have their own license, including the original Qwen Community License; they are not covered by this repository's code license.

The [archived upstream README](docs/UPSTREAM-README.md) describes the original Minirun distribution. Its download/TestFlight links belong to that original product.
