# Experimental Flint streaming engine

The official Recursive Flint INT4 GGUF is the first target. This directory is
an implementation and test boundary, not an enabled or released phone model.
Ornith, QPack/LES, the existing app, API and user data are preserved.

Implemented: official Flint graph/vocabulary backport, bounded shared expert
banks, direct GGUF expert reads, compact SWA/full-attention KV, model admission,
measured loader phases, cancellation/reset/unload, state-only prefill, native
tokenization and publisher chat-template rendering. The opt-in Swift actor now
adds native seeded/greedy generation, incremental UTF-8/reasoning, a strict
Flint tool-call parser and a shared foreground chat/LAN API owner. The app has
an opt-in model selector, separate history/preferences and exact-token 1K/4K/10K
prompt benchmarks. These new Apple integration paths remain uncompiled here.

See [source pins and limitations](PROVENANCE.md) and [model identity](model-profile.json).
The default small-model/vocabulary controls never fetch a complete model.
`download_flint_model.py --output <cache-directory>` is an explicit opt-in,
single-copy, resumable download of the pinned 25,992,966,976-byte public GGUF.
Its full SHA has now been checked locally. No complete trained graph was run
on the Windows host. The original weights, quantization and top-6 stay intact.

`verify_flint_weight_slices.py --model <pinned-gguf> --control <native-binary>
--output <private-directory>` reads only 414,720 weight bytes after full SHA
verification, checking 27 real gate/up/down slices across three layers and
expert IDs 0/127/255. Dequantization matched NumPy exactly; maximum projection
absolute error was 0.00014495849609375 with exact-Q8 activation fixtures.
This is format/operator evidence, not trained output quality or phone speed.

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
state-only prefill parity, no-copy logits, greedy/seeded decoding, bounded real
routing diagnostics, KV append/reset, cancellation and unloading. It
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
# App integration retains LES and adds the isolated Flint product transitively.
FLASH_LOCAL_ENABLE_LES=1 FLASH_LOCAL_ENABLE_FLINT=1 xcodebuild build \
  -project Apps/Minirun/Minirun.xcodeproj -scheme Minirun-iOS \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO
```

The builder checks exact exports, module import, architecture, framework
dependencies and slice hashes. Exported `flf_*` symbols isolate this runtime
from the existing `flo_*` and tokenizer/reference libraries. The Linux build
also restricts its shared-library exports.

The app's new path remains disabled unless `FLASH_LOCAL_ENABLE_FLINT=1` was
present during package resolution; the existing release workflow is unchanged.
The experimental selector shuts down the former owner first. A common engine
lease also rejects concurrent Ornith/Flint native ownership. No second app,
signing identity or production backend replacement was created.

Before generation, `FlintSession.prepare` performs four real CPU/Metal logit
checks and compares every actually computed top-6 set. Final-layer FFNs at
positions without requested logits are correctly omitted from the trace.
Results must satisfy the explicit numeric policy; a discrepancy closes the
candidate. Passed admission is HMAC-bound to the model, host executable,
actual native framework bytes, OS and context/batch configuration. Reload still
checks the verified descriptor and current memory. A cache record is not a
long-context or whole-quality certification.

Still required before release: Apple compile/package tests, real light/dark
iOS/macOS visual review, actual Metal/trained-model admission and a real
model-generated tool call → local result → next model step. Portable Swift
Foundation controls pass; they exclude the native actor and Network.framework.
The newly added TCP readiness test is written but has not run on this host.

Flint uses native reasoning. Ornith's Off/Small/Medium/Ultra/forced-budget
controls are not advertised for it. Capability-aware launcher helpers omit those
legacy defaults, preserving actual reasoning output. The direct phone API
rejects unsupported reasoning controls. OpenCode/Codex compatibility remains a
real-phone acceptance gate; Hermes remains gated below its 64K requirement.

## Phone acceptance after an Apple build

Select **Prova Flint**, download the official GGUF (or retain the app's completed
download), then **Ladda modell**. First admission may be slow because it runs a
real CPU/Metal comparison. Unload/reload should reuse its signed result in the
same build/configuration. Any disagreement must remain a hard stop.

Send two messages in the same chat and save the second report under Prestanda.
Record actual new/reused tokens, prompt time, TTFT, decode speed, footprint,
expert-read bytes and thermal state. The 1K/4K/10K buttons leave chat history
intact and report actual inference; 10K needs an admitted 16K/32K window. A run
ending before 128 output tokens is explicitly not a comparable decode test.
No batch combination is labelled optimal before these phone measurements.

In API, start the server and copy its live address, key and selected client's
commands to the computer. `/health` returns 503 while Flint is unloaded/inactive;
`/v1/models` retains the configured identity. Stop API keeps the model loaded.
Test foreground/background transitions, cancellation and a local fixture tool
before any authorized agent workflow. The iPhone never executes tools.

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
