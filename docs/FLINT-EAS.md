# Optional EAS Apple validation

This is a delivery wrapper for the existing Swift application, on the isolated
`codex/flint-engine` branch. It adds no Expo/React Native runtime and performs no
prebuild, project relocation, bundle ID change, credential migration or signing.
The existing GitHub CI and original app are unchanged.

The linked Expo account must have free iOS builds remaining. Pin EAS CLI 27.1.0
and the `macos-sequoia-15.6-xcode-26.2` medium worker. The Free plan currently
allows 15 iOS builds/month and a 45-minute timeout. Check actual account usage
before dispatch; never upgrade billing or choose a large worker implicitly.

```sh
eas account:usage dhafgard --json --non-interactive
eas build --platform ios --profile flint-native --non-interactive --no-wait
eas build --platform ios --profile flint-compile --non-interactive --no-wait
```

`flint-native` builds symbol-isolated Mac/iPhone/Simulator frameworks, then
executes the small FP64/quantized graph, real vocabulary/template, and native
ownership/sampling/KV/routing controls. It downloads only metadata, never the
26 GB trained model. `flint-compile` adds the existing tokenizer and LES runtimes,
targeted package tests, Mac app compilation and an unsigned physical-iPhone
Release compilation. Models and credentials are excluded by `.easignore`.

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
