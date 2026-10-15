#!/bin/bash
# Temporary credentials on a disposable macOS runner. Never run with shell -x.
set -euo pipefail
umask 077
test "$(uname -s)" = Darwin
: "${RUNNER_TEMP:?}" "${GITHUB_ENV:?}" "${IOS_DISTRIBUTION_P12_PASSWORD:?}"
SIGNING_DIR="$RUNNER_TEMP/flash-local-signing"
mkdir -p "$SIGNING_DIR"
export SIGNING_DIR
python3 - <<'PY'
import base64, os, pathlib, re, uuid
root = pathlib.Path(os.environ['SIGNING_DIR'])
assert re.fullmatch(r'[A-Z0-9]{10}', os.environ['ASC_KEY_ID'])
assert re.fullmatch(r'[A-Z0-9]{10}', os.environ['APPLE_TEAM_ID'])
uuid.UUID(os.environ['ASC_ISSUER_ID'])
for variable, name, limit in [('ASC_PRIVATE_KEY_B64', 'AuthKey.p8', 16384),
                              ('IOS_DISTRIBUTION_P12_B64', 'distribution.p12', 131072)]:
    data = base64.b64decode(os.environ[variable], validate=True)
    assert 0 < len(data) <= limit
    path = root / name
    with path.open('xb') as stream:
        stream.write(data)
    path.chmod(0o600)
with open(os.environ['GITHUB_ENV'], 'a') as stream:
    stream.write('ASC_KEY_PATH=' + str(root / 'AuthKey.p8') + '\n')
PY
KEYCHAIN_PATH="$RUNNER_TEMP/flash-local-signing.keychain-db"
KEYCHAIN_PASSWORD="$(openssl rand -hex 32)"
security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
security import "$SIGNING_DIR/distribution.p12" -P "$IOS_DISTRIBUTION_P12_PASSWORD" \
  -T /usr/bin/codesign -t cert -f pkcs12 -k "$KEYCHAIN_PATH" >/dev/null
security set-key-partition-list -S apple-tool:,apple:,codesign: \
  -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null
security list-keychains -d user -s "$KEYCHAIN_PATH" "$HOME/Library/Keychains/login.keychain-db"
