#!/usr/bin/env bash
# Private MiniCPM TestFlight delivery. Invoke ONLY via approved manual workflow.
set -euo pipefail
umask 077
: "$RUNNER_TEMP" "$GITHUB_RUN_NUMBER"
for name in APPLE_TEAM_ID ASC_KEY_ID ASC_ISSUER_ID ASC_PRIVATE_KEY_B64 IOS_DISTRIBUTION_P12_B64 IOS_DISTRIBUTION_P12_PASSWORD MINICPM_APP_STORE_APP_ID; do
  test -n "$(printenv "$name")" || { echo "Missing required Apple configuration: $name"; exit 1; }
done
[[ "$MINICPM_APP_STORE_APP_ID" =~ ^[0-9]+$ ]]
[[ "$APPLE_TEAM_ID" =~ ^[A-Z0-9]{10}$ ]]
[[ "$ASC_KEY_ID" =~ ^[A-Z0-9]{10}$ ]]
BUNDLE_ID="com.dhafen.minicpmcontext"
UPSTREAM_DIR="$1"
test -d "$UPSTREAM_DIR/MiniCPM-V-demo.xcodeproj"
KEY_DIR="$RUNNER_TEMP/minicpm-signing"
KEYCHAIN="$RUNNER_TEMP/minicpm-signing.keychain-db"
mkdir -p "$KEY_DIR" "$HOME/.appstoreconnect/private_keys"
cleanup() {
  security list-keychains -d user -s "$HOME/Library/Keychains/login.keychain-db" >/dev/null 2>&1 || true
  security delete-keychain "$KEYCHAIN" >/dev/null 2>&1 || true
  rm -rf "$KEY_DIR" "$RUNNER_TEMP/minicpm-private.ipa" "$RUNNER_TEMP/minicpm-archive.xcarchive" "$RUNNER_TEMP/minicpm-export"
  rm -f "$HOME/.appstoreconnect/private_keys/AuthKey_$ASC_KEY_ID.p8"
}
trap cleanup EXIT
python3 - <<'PY'
import base64,os,pathlib,uuid
uuid.UUID(os.environ["ASC_ISSUER_ID"])
path=pathlib.Path(os.environ["RUNNER_TEMP"])/"minicpm-signing"
for key,name in [("ASC_PRIVATE_KEY_B64","AuthKey.p8"),("IOS_DISTRIBUTION_P12_B64","Distribution.p12")]:
    raw=base64.b64decode(os.environ[key],validate=True)
    if not (0<len(raw)<131072): raise RuntimeError(f"Invalid {key} length")
    out=path/name; out.write_bytes(raw); out.chmod(0o600)
PY
KEYCHAIN_PASSWORD="$(openssl rand -hex 32)"
security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
security import "$KEY_DIR/Distribution.p12" -P "$IOS_DISTRIBUTION_P12_PASSWORD" -T /usr/bin/codesign -t cert -f pkcs12 -k "$KEYCHAIN" >/dev/null
security set-key-partition-list -S apple-tool:,apple:,codesign: -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN" >/dev/null
security list-keychains -d user -s "$KEYCHAIN" "$HOME/Library/Keychains/login.keychain-db"
cd "$UPSTREAM_DIR"
MINIMAL_MODE=ios-device ./scripts/build_xcframework.sh
xcodebuild archive \
  -project MiniCPM-V-demo.xcodeproj -scheme MiniCPM-V-demo \
  -configuration Release -destination 'generic/platform=iOS' \
  -archivePath "$RUNNER_TEMP/minicpm-archive.xcarchive" \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" DEVELOPMENT_TEAM="$APPLE_TEAM_ID" \
  CODE_SIGN_STYLE=Automatic CODE_SIGNING_ALLOWED=YES \
  CURRENT_PROJECT_VERSION="$GITHUB_RUN_NUMBER" \
  -allowProvisioningUpdates \
  -authenticationKeyPath "$KEY_DIR/AuthKey.p8" \
  -authenticationKeyID "$ASC_KEY_ID" \
  -authenticationKeyIssuerID "$ASC_ISSUER_ID" \
  2>&1 | tee "$RUNNER_TEMP/minicpm-private-archive.log" | tail -60
INFO="$RUNNER_TEMP/minicpm-archive.xcarchive/Products/Applications/MiniCPM-V-demo.app/Info.plist"
test -f "$INFO"
test "$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$INFO")" = "$BUNDLE_ID"
cat > "$RUNNER_TEMP/minicpm-ExportOptions.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>method</key><string>app-store-connect</string>
<key>destination</key><string>export</string>
<key>signingStyle</key><string>automatic</string>
<key>teamID</key><string>$APPLE_TEAM_ID</string>
<key>manageAppVersionAndBuildNumber</key><false/>
</dict></plist>
PLIST
xcodebuild -exportArchive \
  -archivePath "$RUNNER_TEMP/minicpm-archive.xcarchive" \
  -exportOptionsPlist "$RUNNER_TEMP/minicpm-ExportOptions.plist" \
  -exportPath "$RUNNER_TEMP/minicpm-export" -allowProvisioningUpdates \
  -authenticationKeyPath "$KEY_DIR/AuthKey.p8" \
  -authenticationKeyID "$ASC_KEY_ID" \
  -authenticationKeyIssuerID "$ASC_ISSUER_ID" \
  2>&1 | tee "$RUNNER_TEMP/minicpm-private-export.log" | tail -60
IPA=$(find "$RUNNER_TEMP/minicpm-export" -name '*.ipa' -print -quit)
test -n "$IPA"
cp "$KEY_DIR/AuthKey.p8" "$HOME/.appstoreconnect/private_keys/AuthKey_$ASC_KEY_ID.p8"
chmod 600 "$HOME/.appstoreconnect/private_keys/AuthKey_$ASC_KEY_ID.p8"
xcrun altool --upload-app --type ios --file "$IPA" \
  --apiKey "$ASC_KEY_ID" --apiIssuer "$ASC_ISSUER_ID" \
  2>&1 | tee "$RUNNER_TEMP/minicpm-private-upload.log" | tail -40
echo "Upload complete. Apple build processing and tester permissions are NOT verified here."
