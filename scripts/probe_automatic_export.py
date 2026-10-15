"""Ask Xcode to export locally with automatic provisioning; never upload."""
import argparse
import datetime
import json
import os
import pathlib
import plistlib
import subprocess
import sys


MEMORY_KEYS = ('com.apple.developer.kernel.increased-memory-limit',
               'com.apple.developer.kernel.extended-virtual-addressing')


def main():
    if sys.platform != 'darwin':
        raise RuntimeError('Xcode export requires macOS')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(os.environ['RUNNER_TEMP']) / 'flash-local-signing'
    options = {'method': 'app-store-connect', 'destination': 'export',
               'teamID': os.environ['APPLE_TEAM_ID'], 'signingStyle': 'automatic',
               'signingCertificate': 'Apple Distribution', 'testFlightInternalTestingOnly': True,
               'manageAppVersionAndBuildNumber': False, 'stripSwiftSymbols': True}
    plist = root / 'ExportOptions.plist'
    plist.write_bytes(plistlib.dumps(options))
    command = ['xcodebuild', '-exportArchive', '-archivePath', str(args.archive),
               '-exportPath', str(root / 'export'), '-exportOptionsPlist', str(plist),
               '-allowProvisioningUpdates', '-authenticationKeyPath', os.environ['ASC_KEY_PATH'],
               '-authenticationKeyID', os.environ['ASC_KEY_ID'],
               '-authenticationKeyIssuerID', os.environ['ASC_ISSUER_ID']]
    with (root / 'export.log').open('wb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        # Credential files and the full signing log stay off public artifacts.
        # Apple errors describe the blocker without printing private key data.
        lines = (root / 'export.log').read_text(errors='replace').splitlines()
        errors = [line for line in lines if 'error:' in line.lower() or 'error exportArchive:' in line]
        for line in errors[-12:]:
            print(line[:2000])
        raise SystemExit('Automatic export failed. No build was uploaded.')
    ipas = list((root / 'export').glob('*.ipa'))
    if len(ipas) != 1:
        raise RuntimeError('Expected exactly one locally exported IPA')
    # The unsigned archive cannot prove entitlement preservation. Inspect the
    # actual exported signature and embedded profile before calling this ready.
    import tempfile
    import zipfile
    with tempfile.TemporaryDirectory(dir=root) as temp:
        with zipfile.ZipFile(ipas[0]) as archive:
            for item in archive.infolist():
                if not (pathlib.Path(temp) / item.filename).resolve().is_relative_to(pathlib.Path(temp).resolve()):
                    raise RuntimeError('IPA path escapes extraction root')
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Unexpected symlink in device IPA')
            if sum(item.file_size for item in archive.infolist()) > 500_000_000:
                raise RuntimeError('Unexpected IPA contents size')
            archive.extractall(temp)
            for item in archive.infolist():
                mode = (item.external_attr >> 16) & 0o777
                if mode:
                    (pathlib.Path(temp) / item.filename).chmod(mode)
        app = pathlib.Path(temp) / 'Payload/FlashLocal.app'
        subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
        signature = subprocess.run(['codesign', '-d', '--entitlements', ':-', str(app)],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        entitlements = plistlib.loads(signature.stdout)
        profile = plistlib.loads(subprocess.run(['security', 'cms', '-D', '-i', str(app / 'embedded.mobileprovision')],
                                 check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout)
        team = os.environ['APPLE_TEAM_ID']
        if profile['TeamIdentifier'] != [team] or profile['Entitlements'].get('get-task-allow', False):
            raise RuntimeError('Export did not use the correct distribution team/profile')
        if 'ProvisionedDevices' in profile or profile.get('ProvisionsAllDevices', False):
            raise RuntimeError('TestFlight requires an App Store distribution profile')
        if profile['ExpirationDate'] <= datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None):
            raise RuntimeError('Exported profile expired')
        if entitlements.get('application-identifier') != team + '.com.dhafen.flashlocal':
            raise RuntimeError('Wrong exported application identity')
        if entitlements.get('get-task-allow', False):
            raise RuntimeError('Exported app permits debugging')
        for key in MEMORY_KEYS:
            if entitlements.get(key) is not True or profile['Entitlements'].get(key) is not True:
                raise RuntimeError('Required memory entitlement absent from exported app or profile: ' + key)
    pathlib.Path('ios-signing-probe.json').write_text(json.dumps({
        'bundleIdentifier': 'com.dhafen.flashlocal', 'signatureVerified': True,
        'memoryEntitlementsVerified': True, 'profileType': 'app-store',
        'exportedLocally': True, 'uploadedToTestFlight': False}, indent=2) + '\n', encoding='utf-8')
    print('Automatic distribution export verified both memory entitlements. No build was uploaded.')


if __name__ == '__main__':
    main()
