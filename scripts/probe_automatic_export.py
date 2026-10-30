"""Ask Xcode to export locally with automatic provisioning; never upload."""
import argparse
import datetime
import json
import hashlib
import os
import pathlib
import plistlib
import subprocess
import sys


MEMORY_KEYS = ('com.apple.developer.kernel.increased-memory-limit',
               'com.apple.developer.kernel.extended-virtual-addressing')


def matching_uploads(api, app_id, version, build):
    return api.all('apps/' + app_id + '/buildUploads', query={'filter[cfBundleVersion]': build,
        'filter[cfBundleShortVersionString]': version, 'filter[platform]': 'IOS', 'limit': '200'})


def require_fresh_export_number(api, app_id, version, build):
    from deliver_testflight import find_build
    if find_build(api, app_id, build) is not None or matching_uploads(api, app_id, version, build):
        raise ValueError('Build number became occupied before export')


def capture_export_reservation(api, app_id, version, build, started, ended):
    """Bind only a new, empty Xcode-created reservation to this export.

    The final Apple file checksum and linked build still prove the IPA identity
    before any TestFlight association. Any ambiguous reservation fails closed.
    """
    from apple_api import resource_id
    rows = matching_uploads(api, app_id, version, build)
    if not rows:
        return None  # Some Xcode versions do not reserve during export.
    if len(rows) != 1:
        raise ValueError('Ambiguous upload reservations after export')
    row = rows[0]; attrs = row.get('attributes', {})
    if (attrs.get('cfBundleVersion'), attrs.get('cfBundleShortVersionString'), attrs.get('platform')) != (build, version, 'IOS'):
        raise ValueError('Xcode reservation identity differs from export')
    if attrs.get('state', {}).get('state') != 'AWAITING_UPLOAD':
        raise ValueError('Xcode reservation is not empty and awaiting upload')
    try:
        created = datetime.datetime.fromisoformat(attrs['createdDate'].replace('Z', '+00:00'))
        if created.tzinfo is None:
            raise ValueError('Missing timezone')
        created = created.astimezone(datetime.timezone.utc)
    except (KeyError, ValueError, TypeError):
        raise ValueError('Xcode reservation has no trustworthy creation time') from None
    # Account for small service/runner clock skew without accepting stale rows.
    margin = datetime.timedelta(seconds=60)
    if not started - margin <= created <= ended + margin:
        raise ValueError('Xcode reservation was not created during this export')
    identifier = resource_id(row.get('id'))
    if api.all('buildUploads/' + identifier + '/buildUploadFiles', query={'limit': '200'}):
        raise ValueError('Xcode reservation already contains files')
    return identifier


def export_options(team, profile_uuid=None):
    options = {'method': 'app-store-connect', 'destination': 'export',
               'teamID': team, 'signingStyle': 'manual' if profile_uuid else 'automatic',
               'signingCertificate': 'Apple Distribution', 'testFlightInternalTestingOnly': True,
               'manageAppVersionAndBuildNumber': False, 'stripSwiftSymbols': True}
    if profile_uuid:
        import uuid
        uuid.UUID(profile_uuid)
        options['provisioningProfiles'] = {'com.dhafen.flashlocal': profile_uuid}
    return options


def export_command(archive, root, options_path, environment, manual=False):
    command = ['xcodebuild', '-exportArchive', '-archivePath', str(archive),
               '-exportPath', str(root / 'export'), '-exportOptionsPlist', str(options_path)]
    if not manual:
        command += ['-allowProvisioningUpdates', '-authenticationKeyPath', environment['ASC_KEY_PATH'],
                    '-authenticationKeyID', environment['ASC_KEY_ID'],
                    '-authenticationKeyIssuerID', environment['ASC_ISSUER_ID']]
    return command


def main():
    if sys.platform != 'darwin':
        raise RuntimeError('Xcode export requires macOS')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=pathlib.Path)
    parser.add_argument('--profile', type=pathlib.Path, help='Use a prevalidated existing App Store profile without an API key')
    args = parser.parse_args()
    root = pathlib.Path(os.environ['RUNNER_TEMP']) / 'flash-local-signing'
    requested_path = pathlib.Path(__file__).resolve().parents[1] / 'Apps/Minirun/Minirun-iOS.entitlements'
    requested = plistlib.loads(requested_path.read_bytes())
    if set(requested) != set(MEMORY_KEYS) or not all(requested[key] is True for key in MEMORY_KEYS):
        raise RuntimeError('Unexpected app capability request')
    app_in_archive = args.archive / 'Products/Applications/FlashLocal.app'
    # CODE_SIGNING_ALLOWED=NO leaves no entitlement blob for the exporter to
    # discover. Stage an ad-hoc signature with only the app's requested flags;
    # Xcode must still obtain an authorized Apple distribution profile and sign
    # the exported IPA. An ad-hoc signature is never accepted as final proof.
    subprocess.run(['codesign', '--force', '--sign', '-', '--generate-entitlement-der',
                    '--entitlements', str(requested_path), str(app_in_archive)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    staged = plistlib.loads(subprocess.run(['codesign', '-d', '--entitlements', ':-', str(app_in_archive)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout)
    if not all(staged.get(key) is True for key in MEMORY_KEYS):
        raise RuntimeError('Requested memory capabilities did not enter the staged archive')
    profile_uuid = None
    if args.profile:
        profile = plistlib.loads(subprocess.run(['security', 'cms', '-D', '-i', str(args.profile)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout)
        if (profile.get('TeamIdentifier') != [os.environ['APPLE_TEAM_ID']]
                or profile.get('Entitlements', {}).get('application-identifier') != os.environ['APPLE_TEAM_ID'] + '.com.dhafen.flashlocal'
                or 'ProvisionedDevices' in profile or profile.get('ProvisionsAllDevices', False)
                or profile.get('Entitlements', {}).get('get-task-allow', False)
                or any(profile.get('Entitlements', {}).get(key) is not True for key in MEMORY_KEYS)):
            raise RuntimeError('Manual profile does not authorize this exact distribution app and memory capabilities')
        profile_uuid = profile['UUID']
    options = export_options(os.environ['APPLE_TEAM_ID'], profile_uuid)
    plist = root / 'ExportOptions.plist'
    plist.write_bytes(plistlib.dumps(options))
    command = export_command(args.archive, root, plist, os.environ, manual=bool(args.profile))
    reservation_api = None
    reservation_id = None
    if os.environ.get('FLASH_LOCAL_CAPTURE_XCODE_RESERVATION') == '1':
        from apple_api import AppleAPI, resource_id
        from deliver_testflight import private_target
        archive_info = plistlib.loads((app_in_archive / 'Info.plist').read_bytes())
        planned_version, planned_build = archive_info['CFBundleShortVersionString'], archive_info['CFBundleVersion']
        if planned_build != os.environ['FLASH_LOCAL_BUILD_NUMBER']:
            raise ValueError('Archive build differs from the preflight-selected number')
        app_id = resource_id(os.environ['APP_STORE_APP_ID'])
        reservation_api = AppleAPI(os.environ['ASC_KEY_PATH'], os.environ['ASC_KEY_ID'], os.environ['ASC_ISSUER_ID'])
        private_target(reservation_api, app_id, os.environ['TESTFLIGHT_INTERNAL_GROUP_ID'],
                       json.loads(os.environ['TESTFLIGHT_ALLOWED_TESTER_IDS']))
        require_fresh_export_number(reservation_api, app_id, planned_version, planned_build)
    export_started = datetime.datetime.now(datetime.timezone.utc)
    with (root / 'export.log').open('wb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    export_ended = datetime.datetime.now(datetime.timezone.utc)
    if result.returncode:
        # Credential files and the full signing log stay off public artifacts.
        # Apple errors describe the blocker without printing private key data.
        lines = (root / 'export.log').read_text(errors='replace').splitlines()
        errors = [line for line in lines if 'error:' in line.lower() or 'error exportArchive:' in line]
        for line in errors[-12:]:
            print(line[:2000])
        raise SystemExit('Automatic export failed. No build was uploaded.')
    if reservation_api is not None:
        reservation_id = capture_export_reservation(reservation_api, app_id, planned_version, planned_build,
                                                     export_started, export_ended)
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
        app_info = plistlib.loads((app / 'Info.plist').read_bytes())
        if app_info.get('CFBundleIdentifier') != 'com.dhafen.flashlocal':
            raise RuntimeError('Wrong IPA bundle identifier')
        if app_info.get('ITSAppUsesNonExemptEncryption') is not False:
            raise RuntimeError('Explicit reviewed encryption exemption is missing')
        from check_ios_archive import privacy_manifest
        privacy_manifest(app / 'PrivacyInfo.xcprivacy')
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
        observed = {key: {'signature': entitlements.get(key) is True,
                         'profile': profile['Entitlements'].get(key) is True} for key in MEMORY_KEYS}
        print('Exported memory capability admission: ' + json.dumps(observed, sort_keys=True))
        if not all(all(status.values()) for status in observed.values()):
            raise RuntimeError('Required memory entitlement absent from exported app or profile')
    digest = hashlib.sha256()
    with ipas[0].open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): digest.update(chunk)
    pathlib.Path('ios-signing-probe.json').write_text(json.dumps({
        'bundleIdentifier': 'com.dhafen.flashlocal', 'signatureVerified': True,
        'memoryEntitlementsVerified': True, 'profileType': 'app-store',
        'privacyManifestVerified': True,
        'exportedLocally': True, 'uploadedToTestFlight': False, 'internalTestingOnly': True,
        'ipaSHA256': digest.hexdigest(), 'sourceRevision': os.environ.get('GITHUB_SHA', 'unknown'),
        'workflowRunID': os.environ.get('GITHUB_RUN_ID'),
        'usesNonExemptEncryption': False,
        'xcodeReservedUploadID': reservation_id, 'xcodeReservationVerified': reservation_id is not None,
        'version': app_info['CFBundleShortVersionString'], 'build': app_info['CFBundleVersion']}, indent=2) + '\n', encoding='utf-8')
    print('Automatic distribution export verified both memory entitlements. No build was uploaded.')


if __name__ == '__main__':
    main()
