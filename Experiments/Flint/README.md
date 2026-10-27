# Experimental Flint streaming engine

The official Recursive Flint INT4 GGUF is the first target. This directory is
an implementation and test boundary, not an enabled or released phone model.
Ornith, QPack/LES, the existing app, API and user data are preserved.

Implemented: official Flint graph/vocabulary backport, bounded shared expert
banks, direct GGUF expert reads, compact SWA/full-attention KV, model admission,
measured loader phases, cancellation/reset/unload, state-only prefill, native
tokenization and publisher chat-template rendering. The opt-in Swift actor
adapter keeps native ownership independent of views and checks exact prefixes.

See [source pins and limitations](PROVENANCE.md) and [model identity](model-profile.json).
No complete trained model is downloaded by the control scripts.

## Portable native checks

Prepare once from the repository root:

```sh
python scripts/prepare_flint_runtime.py
cmake -S Experiments/Flint/Native -B .work/flint-native-build \
  -DOMNI_SOURCE="$PWD/.work/flint-source" -DCMAKE_BUILD_TYPE=Release
cmake --build .work/flint-native-build --parallel 3
python -m pip install -r Experiments/Flint/reference-requirements.txt
python scripts/verify_flint_controls.py \
  --control .work/flint-native-build/bin/flint-control \
  --quantizer .work/flint-native-build/bin/flint-quantize-control
python scripts/fetch_flint_metadata.py
python scripts/verify_flint_protocol.py \
  --prefix .work/flint-inputs/gguf-prefix.bin \
  --tokenizer .work/flint-inputs/tokenizer.json \
  --native-bin .work/flint-native-build/bin
python -m unittest discover -s scripts/tests
```

On Windows, run CMake in Ubuntu WSL with the source path under `/mnt/c/...` and
keep build outputs in the Linux home directory (WSL may clear `/tmp`). The two
verification scripts also accept `--wsl Ubuntu`, a Linux `--control`/
`--quantizer`/`--native-bin`, and Windows input/output paths. Use an isolated
Python environment for the pinned reference dependencies.

FP32 tests compare every logit and top-6 decision with independent NumPy/FP64,
through SWA rollover at normal and large activations. Q4_1 tests use GGML's
canonical encoder, then compare resident-kernel and streamed-kernel execution.
They do not compare dequantized FP64 with CPU Q8 activation arithmetic as if
those were identical precision policies. Each input remains under 64 MiB.

`flint-api-control small.gguf empty-private-directory` tests one context,
state-only prefill parity, KV append/reset, cancellation and unloading. It
rejects model files over 64 MiB. The library API has separate verified-file
authority and memory admission for the real checkpoint.

## Apple checks and remaining integration

On an Apple-silicon Mac with Xcode:

```sh
python scripts/build_flint_runtime.py
# Also prepare the existing tokenizer XCFramework as required by Package.swift.
python scripts/build_tokenizer_xcframework.py
FLASH_LOCAL_ENABLE_FLINT=1 xcodebuild -scheme FlintEngine \
  -destination 'platform=macOS,arch=arm64' test
```

The builder checks exact exports, module import, architecture, framework
dependencies and slice hashes. Exported `flf_*` symbols isolate this runtime
from the existing `flo_*` and tokenizer/reference libraries. The Linux build
also restricts its shared-library exports.

Still required before enabling chat/API: compile/test the Swift adapter and
Metal path, connect a shared app-level engine lease/model selector, add the
Flint XML tool-call parser with existing schema validation, and compare full
trained CPU/Metal logits/routing before enabling generation. Preserve the
publisher's reasoning replay; do not assume Ornith's forced thinking controls
are equivalent. Then test actual tool call → local result → next model step.

The app owner must cancel GPU work before iOS background suspension while
retaining the model across ordinary navigation. A retained actor does not grant
background execution or keep a suspended LAN server reachable.

Start phone checks at 8K. Record 128+ output tokens, cold/warm app cache, prompt
sizes 1K/4K/10K, physical batch 1/2/4/8, real footprint, I/O, Metal timestamps,
thermal state and prefix reuse. Compare with Ornith on the same phone/prompts.
Advance to 32K and 65K only with actual admission, context-quality and stability
evidence. No trained decode speed, TTFT or usable long-context claim exists yet.

The current Windows host cannot run Xcode/Metal. The account's free GitHub
Actions minutes are exhausted (verified 27 October 2026). No paid runner,
workflow/signing change, archive or TestFlight upload was attempted here.
