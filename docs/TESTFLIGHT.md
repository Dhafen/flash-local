# Private internal TestFlight delivery

The real arm64 iPhone Release archive and Apple distribution export passed in
run `38740774972` at `eebae31`. The exported signature and App Store profile both
contained increased-memory-limit and extended-virtual-addressing. The unsigned
app was about 28 MB; model weights were absent. No IPA was uploaded in that run.

The app record and an internal group containing only the verified account holder
are configured. The upload path remains unverified against an actual upload.
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
in the actual app bundle, before signing/upload. See Apple's
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
56.1 GB offline inference, phone RAM/throughput, long context or model quality.

Apple's upload and group-association documentation:
https://developer.apple.com/help/app-store-connect/manage-builds/upload-builds/
https://developer.apple.com/documentation/appstoreconnectapi/post-v1-betagroups-_id_-relationships-builds
