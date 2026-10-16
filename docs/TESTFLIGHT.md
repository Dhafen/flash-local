# Private internal TestFlight delivery

Latest version **0.1.0 (84)**, source
`b093bec6b3d4061b6693e07fdb3e4a6ab9a95606`, is available to the configured
own internal tester. [Full run 38946605484](https://github.com/Dhafen/flash-local/actions/runs/38946605484)
passed all gates, including 569 app tests, the package suite (474 tests, nine
environment-gated skips), synthetic reference/pairing checks, twelve reviewed
light/dark Mac/iOS captures, actual device archive/signing and private upload.
A fresh read-only Apple audit confirms `VALID`, `INTERNAL_ONLY`,
`IN_BETA_TESTING`, not expired, the exact private group and unchanged sole-owner
membership. The sealed IPA SHA-256 is
`9e45fb5ccb73d93380eade985c935821b60823bfa6df4b3d939b3b044bb26755`.
The paired gate/up path preserves all logit bits in both tiny graph controls.
It needs no model redownload; its actual phone gain has not been measured.
The isolated Swiftlet/QPack experiment is not included in build 84.

## Build 80 baseline and previous delivery

Version **0.1.0 (80)** from app source
`c17b9cdc6aefd6c5471b69c1c321e7bbe00807a4` is available to the configured
own internal tester. [Full run 38926302018](https://github.com/Dhafen/flash-local/actions/runs/38926302018)
concluded **success**, including all test/build gates, Simulator/layout review,
device Release archive, actual Apple signing and private upload. Upload exited
zero. A fresh read-only Apple audit confirms `VALID`, `INTERNAL_ONLY`,
`IN_BETA_TESTING`, not expired, exactly the existing private group and unchanged
sole-owner membership. No public link, additional tester or automatic access
to all builds was enabled.

This is the first delivered [Ornith implementation](ORNITH-IMPLEMENTATION.md):
the official pinned 21.7 GB Q4_K_M model is primary, with a separate chat and
model choice retaining REAM. It includes the strict role-history recovery,
protected dense/expert cache and bounded canonical K/V kernels. The initial
context policy is 4,096 tokens inside the configured 5.5 GB process ceiling.
The unsigned arm64 app was 28,376,940 bytes without weights. Actual Apple
signature/profile, both memory entitlements, privacy manifest and model notices
were verified. The final IPA SHA-256 is
`936e4c2ef6ffd57779bfc9d596896ac0fd06dacb27ecfb5fb08d90805aceaf13`.

Update the existing app without uninstalling and download Ornith separately
through its download button. The trained model was not downloaded on the
development host or run in hosted CI. The first user-provided build 80 phone
report measures 1.493 decode forwards/s, 16.105 seconds for 14 prompt tokens,
zero prompt reuse and 3.872 GB sampled peak. It does not verify sustained speed,
offline execution, full-model numerical agreement, long context or comparative
capability. The next follow-up is a matched build 84 foreground conversation,
then the local numerical diagnostic and offline run.

Earlier candidate 77 at `63a07a043f39759116f6bd74b68e7a765616620f` passed its
test/build/signing gates in
[run 38905646842](https://github.com/Dhafen/flash-local/actions/runs/38905646842),
but an Apple private-group GET timed out before upload. It was not delivered;
a subsequent read-only check confirmed no build 77 record. Source `c17b9cd`
adds bounded retries for socket failures on GET only, without repeating
uncertain writes; all 32 local Python tests pass. The successful full run 80
then rebuilt, signed, uploaded and confirmed its own source-bound artifact.

## Previous delivery: build 68

Version **0.1.0 (68)** from app source
`2db039fd6068fd994142c87fc9392b1a4f1c4e8d` is available to the configured own
internal tester. [Full run 38888291457](https://github.com/Dhafen/flash-local/actions/runs/38888291457)
concluded **success**, including the test gates, Simulator review, device
Release archive, actual Apple distribution export and upload/availability
confirmation. A subsequent read-only Apple audit reconfirmed `VALID`,
`INTERNAL_ONLY`, `IN_BETA_TESTING`, exactly the configured private group and
unchanged sole account-holder membership. No public link, additional tester
or automatic access to all builds was enabled.

The unsigned arm64 device app was 28,270,277 bytes without model payload.
Both memory entitlements and the privacy manifest were verified in the
Apple-signed export. The final IPA SHA-256 is
`ca70651ba64e79a64f776beb8ade6996ef888e7880e505e47f865e63abb6b464`.
The archive report intentionally describes the unsigned intermediate; the
export seal confirms signing, and the separate delivery report confirms
upload and availability. Those stages must not be conflated.

This update contains [foreground prefix retention, bounded parallel reads,
whole selected experts and batched GPU evaluation](REAM-PERFORMANCE.md)
and keeps the same model filename/hash. Update through TestFlight without
uninstalling the app or downloading another model copy. The user's build 63
report measures 0.211 decode forwards/s and a
4.633 GB sampled peak footprint. Speed is unusable despite the short run
fitting the RAM budget. Trained-model numerical/offline/long-context proof
remains outstanding, as does build 68 phone speed. TestFlight availability
does not establish those results. To measure prefix retention, send two short
messages in succession without leaving the app and save the second report.

## Previous delivery: build 63

Version **0.1.0 (63)** from app source
`ddfa7526d8291f7a4b705b985a36da025240626f` passed the full
[run 38877522212](https://github.com/Dhafen/flash-local/actions/runs/38877522212),
including actual signing and verified private TestFlight availability. Its
unsigned device app was 28,234,349 bytes, and the IPA SHA-256 was
`925affddeb1645bbb549fe34c257c76ef99ea10593621259e325b2ba9d954b1e`.
It introduced the packed Metal kernels and separate dense cache, but the
subsequent physical measurement established unusable speed. Build 68 changes
scheduling/session reuse while retaining exactly the same model file.

Candidate build 66 at `229773b` passed core, reference, app and package tests
in run `38885484439` before intentional cancellation ahead of upload to include
whole-expert reads. It was superseded by build 68 and was never delivered.

## Previous delivery: build 56

Version **0.1.0 (56)** from app source `5b602d2` was built, signed and uploaded
by run `38791263363`. All preceding test/build gates passed. The exported Apple
signature and App Store profile contained both increased-memory-limit and
extended-virtual-addressing; its privacy manifest was verified. The unsigned
app was 28,179,909 bytes with no model payload. The uploaded IPA SHA-256 is
`445849a1f69b14e7ebedb62b9a55d829c803c62bcafd79a3c64518e0ec460c26`.

Apple processing reached `VALID` and `INTERNAL_ONLY`. The run then failed on
an unsupported GET of the build's beta-group relationship. The documented
`GET /v1/betaGroups?filter[builds]=...` fixed that query in `f78d164`, with
19 passing local delivery tests. Using that helper against the existing build
completed the private group association without another upload. Apple then
confirmed `IN_BETA_TESTING`, exactly the configured group, and unchanged sole
account-holder membership. The original CI conclusion remains failure; the
separately recorded API recovery is the delivery evidence. No other testers,
public links or automatic access to all builds were enabled.

Apple requires the app record to be created on its website, not through the
apps REST resource:
https://developer.apple.com/documentation/appstoreconnectapi/apps

## Account setup

Create an iOS record for `com.dhafen.flashlocal`. Configure repository variables
`APPLE_TEAM_ID`, `APP_STORE_APP_ID`, `TESTFLIGHT_INTERNAL_GROUP_ID` and
`TESTFLIGHT_ALLOWED_TESTER_IDS`. The last value is a JSON array containing exactly
the intended own internal beta tester's resource ID. Do not add other testers
or create a public link. Every beta group for this app must have public links
and automatic access to all builds disabled.

Apple returns null public-link fields for internal groups, where public links
do not apply. The delivery guard accepts that only for a confirmed internal
group with no public-link URL; external or ambiguous access still refuses.
The internal tester is verified against the account holder for this app's
group, rather than reusing a tester resource ID from a different app.

Repository secrets are `ASC_KEY_ID`, `ASC_ISSUER_ID`, `ASC_PRIVATE_KEY_B64`,
`IOS_DISTRIBUTION_P12_B64` and `IOS_DISTRIBUTION_P12_PASSWORD`. Values, profiles,
keys, signing identities and account details never belong in Git. Temporary
credentials and the IPA stay under the runner's private temporary directory.

## Workflow modes

`regenerate_project_only=true` runs the pinned XcodeGen and publishes the
generated project, schemes and Info plists for review/commit from Windows.
It refuses signing/upload/provisioning flags and runs no app tests. Normal CI
still requires the committed project to match that generator.

Normal CI runs all gates. The separate `check_delivery_tools=true` mode checks
Xcode's altool API-key interface and Python delivery tests without credentials
or uploading. That mode passed in run `38744397286` at `c06f392`.

`provisioning_only=true` compiles the actual device Release app and probes its
signature/profile without package/app tests or Simulator coverage. It requires
`release_archive=true` and `probe_automatic_signing=true`; it cannot deliver.

To deliver after account configuration:

```sh
gh workflow run ci.yml --ref ream-streaming \
  -f release_archive=true -f probe_automatic_signing=true \
  -f upload_private_testflight=true
```

The build number is the workflow run number. A rerun keeps its source and build
number, observes any existing Apple record, and does not blindly re-upload.
Delivery runs have their own non-cancelling concurrency group.

## What delivery verifies

The app bundles `PrivacyInfo.xcprivacy`. File integrity checks use app-container
metadata (`C617.1`), measurements use elapsed event durations (`35F9.1`), model
downloads check space before writing (`E174.1`), and app settings use private
user defaults (`CA92.1`). No tracking or automatic diagnostic collection is
declared. The archive gate verifies the manifest and these four audited reasons
in both the archived app and the final exported IPA before upload. See Apple's
[required reason API documentation](https://developer.apple.com/documentation/bundleresources/app-privacy-configuration/nsprivacyaccessedapitypes/nsprivacyaccessedapitype).

All package/app/reference gates and the Simulator review precede the Release
archive. The archive checker verifies device Mach-O, identity, storage access,
model-license bytes, size and absence of model/key files. An intermediate
ad-hoc signature preserves the two requested app entitlements for Xcode's
automatic distribution export; only the final Apple signature/profile count
as signing evidence.

The export seal binds the final IPA SHA-256 to the exact CI source, run, bundle
and version. The uploader checks this seal, the configured app, the single
allowlisted tester and the private internal group before invoking altool.
After upload it waits for Apple's `VALID` processing state and
`INTERNAL_ONLY` audience, associates only the configured group, and verifies
that group's access plus `IN_BETA_TESTING`. A transport/observation timeout is
not treated as proof that Apple rejected the upload. Uncertain mutations are
not retried automatically.

Only small JSON reports become CI artifacts. The IPA, credentials and private
upload logs are always removed. `testflight-delivery.json` distinguishes upload,
processing and confirmed availability; none of these proves installation,
trained-model offline inference, phone RAM/throughput, long context or model quality.

Apple's upload and group-association documentation:
https://developer.apple.com/help/app-store-connect/manage-builds/upload-builds/
https://developer.apple.com/documentation/appstoreconnectapi/post-v1-betagroups-_id_-relationships-builds
https://developer.apple.com/documentation/appstoreconnectapi/get-v1-betagroups
