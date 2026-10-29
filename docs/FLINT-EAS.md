# Optional EAS Apple validation

This is a delivery wrapper for the existing Swift application, on the isolated
`codex/flint-engine` branch. It adds no Expo/React Native runtime and performs no
prebuild, project relocation, bundle ID change, credential migration or signing.
The existing GitHub CI and original app are unchanged.

EAS CLI 27.1.0 classifies a project outside `ios/` as managed while resolving
metadata, even with a custom build. Its entitlement introspector requires the
pinned **development-only** `expo-modules-autolinking` package. This is only a
local CLI metadata dependency: no Expo/React Native app runtime, prebuild or
generated iOS project is used. Run `npm ci --ignore-scripts` before EAS commands.
Windows can archive tracked symlinks as text. The Apple bootstrap restores only
Git-declared links whose unchanged target text points to a file inside this
source tree; modified content and escaping targets are rejected before repair.

The linked Expo account must have free iOS builds remaining. Pin EAS CLI 27.1.0
and the `macos-sequoia-15.6-xcode-26.2` medium worker. The Free plan currently
allows 15 iOS builds/month and a 45-minute timeout. Check actual account usage
before dispatch; never upgrade billing or choose a large worker implicitly.

```sh
eas account:usage dhafgard --json --non-interactive
eas build --platform ios --profile flint-native --non-interactive --no-wait
eas build --platform ios --profile flint-compile --non-interactive --no-wait
eas build --platform ios --profile flint-visual --non-interactive --no-wait
```

`flint-native` builds symbol-isolated Mac/iPhone/Simulator frameworks, then
executes the small FP64/quantized graph, real vocabulary/template, and native
ownership/sampling/KV/routing controls. It downloads only metadata, never the
26 GB trained model. `flint-compile` adds the existing tokenizer and LES runtimes,
targeted package tests, Mac app compilation and an unsigned physical-iPhone
Release compilation. Models and credentials are excluded by `.easignore`.
`flint-visual` also captures the three unloaded Flint panes in both appearances
on Mac and Simulator, alongside the four existing visual suites. Its gate
requires actual capture markers; skipped/empty tests cannot silently pass.
Inspect the exported PNGs before treating visual review as complete.

EAS build `7290041a-82a5-442d-957c-e377115f9d17` compiled the Mac app and
package test bundles, then failed before executing XCTest because the worker's
`com.apple.testmanagerd.control` service was unavailable. Device compilation and
Simulator controls now precede host XCTest, and the ephemeral worker runs
Apple's normal `xcodebuild -runFirstLaunch` setup with bounded diagnostics.
Tests run serially. No security settings are disabled, and a missing service
still fails the tests and prevents delivery; initialization is not test proof.
The next worker audit confirmed its steps run in the `System` bootstrap while
Apple's `com.apple.testmanagerd` is already registered in `gui/501`. Mac XCTest
therefore enters the existing user session via `launchctl asuser` and drops
back to that same unprivileged user before launching Xcode. A real `Aqua`
bootstrap is required; unrelated credentials are not forwarded. Empty/skipped
suite selections cannot replace executed tests or the local TCP round trip.
Hash-bound LF notices are also kept as LF in Windows-made source archives;
the original Qwen license intentionally retains its published CRLF bytes.

Both profiles preserve full logs and a source-bound JSON record in their build
artifacts, including on failure. A green job is only evidence for its listed
checks: it is **not** proof of trained inference, a signed IPA, TestFlight,
phone performance or completion of the existing QPack/LES/parity/visual release
gates. Those still have to pass before signing and publishing.

The EAS upload must contain a clean Git commit; the actual host's revision is
checked against EAS metadata. Native build logs/manifest hashes are retained.
There is deliberately no EAS Submit profile until the required Apple checks and
new Flint UI review pass. Use the existing app identity and authorized signing
resources only when that later delivery step is ready.
