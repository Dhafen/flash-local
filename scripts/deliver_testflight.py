"""Upload only a sealed, verified IPA to one explicitly configured internal group."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

from apple_api import AppleAPI, resource_id

BUNDLE = 'com.dhafen.flashlocal'


def check_tools():
    result = subprocess.run(['xcrun', 'altool', '--help'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    help_text = result.stdout.decode(errors='replace')
    required = ['--upload-app', '--apiKey', '--apiIssuer']
    if not all(flag in help_text for flag in required) or 'API_PRIVATE_KEYS_DIR' not in help_text:
        raise RuntimeError('Installed altool does not document the required API-key upload interface')
    report = {'apiKeyUploadFlagsVerified': True, 'privateKeyDirectoryDocumented': True,
              'uploadAttempted': False}
    pathlib.Path('testflight-tools-check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


def verified_export(ipa, seal_path, source, build, run_id):
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    if not re.fullmatch(r'[0-9a-f]{40}', source) or not re.fullmatch(r'[1-9][0-9]*', build):
        raise ValueError('Delivery requires exact CI source and monotonically numbered build')
    for key in ['signatureVerified', 'memoryEntitlementsVerified', 'privacyManifestVerified', 'exportedLocally', 'internalTestingOnly']:
        if seal.get(key) is not True: raise ValueError('Unverified export: ' + key)
    if seal.get('usesNonExemptEncryption') is not False:
        raise ValueError('Encryption declaration is not verified')
    if (seal.get('bundleIdentifier'), seal.get('profileType'), seal.get('sourceRevision'),
        str(seal.get('build')), str(seal.get('workflowRunID'))) != (BUNDLE, 'app-store', source, build, run_id):
        raise ValueError('Export identity/source/build does not match this delivery run')
    if not 0 < ipa.stat().st_size <= 500_000_000: raise ValueError('Unexpected IPA size')
    digest = hashlib.sha256()
    with ipa.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): digest.update(chunk)
    if digest.hexdigest() != seal.get('ipaSHA256'): raise ValueError('IPA changed after signature validation')
    return seal


def private_target(api, app_id, group_id, allowed):
    resource_id(app_id); resource_id(group_id)
    if not isinstance(allowed, list) or len(allowed) != 1:
        raise ValueError('Private delivery requires exactly the configured own tester')
    for tester in allowed: resource_id(tester)
    app = api.request('apps/' + app_id)['data']
    if app['attributes']['bundleId'] != BUNDLE: raise ValueError('Wrong App Store app record')
    groups = api.all('apps/' + app_id + '/betaGroups', query={'limit': '200'})
    target = None
    for group in groups:
        attrs = group['attributes']
        # Apple returns null for public-link fields on internal groups: those
        # attributes apply only to external testing. Require an internal group
        # and no URL before interpreting null as "not applicable".
        links_disabled = attrs.get('publicLinkEnabled') is False or (
            attrs.get('isInternalGroup') is True and attrs.get('publicLinkEnabled') is None
            and attrs.get('publicLink') is None)
        if not links_disabled or attrs.get('hasAccessToAllBuilds') is not False:
            raise ValueError('App has public or automatically distributed beta access')
        if group['id'] == group_id: target = group
    if target is None or target['attributes'].get('isInternalGroup') is not True:
        raise ValueError('Configured group is not this app\'s internal group')
    testers = api.all('betaGroups/' + group_id + '/relationships/betaTesters', query={'limit': '200'})
    if {item['id'] for item in testers} != set(allowed):
        raise ValueError('Private group membership differs from the own-tester allowlist')


def find_build(api, app_id, build_number):
    builds = api.all('builds', query={'filter[app]': app_id, 'filter[version]': build_number, 'limit': '5'})
    if len(builds) > 1: raise ValueError('Build number is ambiguous across prerelease versions')
    return builds[0] if builds else None


def upload(ipa, key_path, key_id, issuer):
    check_tools()
    directory = key_path.parent
    target = directory / ('AuthKey_' + key_id + '.p8')
    with target.open('xb') as stream: stream.write(key_path.read_bytes())
    target.chmod(0o600)
    environment = dict(os.environ, API_PRIVATE_KEYS_DIR=str(directory))
    # stdout/stderr can include Apple's private delivery details. Keep the full
    # log beside the temporary key and IPA; the workflow always removes them.
    with (directory / 'upload.log').open('wb') as log:
        try:
            result = subprocess.run(['xcrun', 'altool', '--upload-app', '--type', 'ios', '--file', str(ipa),
                '--apiKey', key_id, '--apiIssuer', issuer, '--output-format', 'json'],
                env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        except subprocess.TimeoutExpired:
            print('Upload observation timed out; checking Apple for the same build before any retry.')
            return None
    print('Apple upload command exit code:', result.returncode)
    return result.returncode


def wait_valid(api, app_id, number, version, deadline):
    last_state = None
    while time.monotonic() < deadline:
        build = find_build(api, app_id, number)
        state = build['attributes'].get('processingState') if build else 'AWAITING_RECORD'
        if state != last_state:
            print('Apple processing:', state); last_state = state
        if build:
            attrs = build['attributes']
            if attrs.get('expired') or state in ['FAILED', 'INVALID']:
                raise RuntimeError('Apple build is expired or processing failed: ' + str(state))
            if state == 'VALID':
                resource_id(build['id'])
                release = api.request('builds/' + build['id'] + '/preReleaseVersion')['data']['attributes']
                if release.get('version') != version or release.get('platform') != 'IOS':
                    raise ValueError('Processed build has the wrong prerelease version/platform')
                if attrs.get('buildAudienceType') != 'INTERNAL_ONLY':
                    raise ValueError('Apple did not classify this build as internal-only')
                return build
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    raise TimeoutError('Apple processing is not verified within this run; do not re-upload blindly')


def attach_and_confirm(api, build, group_id, deadline):
    identifier = resource_id(build['id'])
    # Apple does not allow GET on builds/{id}/relationships/betaGroups.
    # List beta groups using the documented build filter instead. This also
    # observes every group, so unexpected access still refuses association.
    query = {'filter[builds]': identifier, 'limit': '200'}
    groups = {item['id'] for item in api.all('betaGroups', query=query)}
    if groups - {group_id}: raise ValueError('Build already has unexpected group access')
    if group_id not in groups:
        api.request('betaGroups/' + group_id + '/relationships/builds', method='POST',
            body={'data': [{'type': 'builds', 'id': identifier}]})
    while time.monotonic() < deadline:
        groups = {item['id'] for item in api.all('betaGroups', query=query)}
        if groups - {group_id}: raise ValueError('Unexpected group access appeared after association')
        detail = api.request('builds/' + identifier + '/buildBetaDetail')['data']['attributes']
        state = detail.get('internalBuildState')
        if groups == {group_id} and state == 'IN_BETA_TESTING': return state
        if state in ['EXPIRED', 'PROCESSING_EXCEPTION', 'MISSING_EXPORT_COMPLIANCE']:
            raise RuntimeError('Apple internal testing is blocked: ' + str(state))
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    raise TimeoutError('Build/group availability has not been confirmed by Apple')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-tools', action='store_true')
    args = parser.parse_args()
    if args.check_tools:
        check_tools(); return
    if sys.platform != 'darwin': raise RuntimeError('Delivery requires the pinned macOS/Xcode CI job')
    root = pathlib.Path(os.environ['RUNNER_TEMP']) / 'flash-local-signing'
    ipas = list((root / 'export').glob('*.ipa'))
    if len(ipas) != 1: raise ValueError('Exactly one verified IPA is required')
    number = os.environ['GITHUB_RUN_NUMBER']
    seal = verified_export(ipas[0], pathlib.Path('ios-signing-probe.json'), os.environ['GITHUB_SHA'],
                           number, os.environ['GITHUB_RUN_ID'])
    app_id = resource_id(os.environ['APP_STORE_APP_ID'])
    group_id = resource_id(os.environ['TESTFLIGHT_INTERNAL_GROUP_ID'])
    allowed = json.loads(os.environ['TESTFLIGHT_ALLOWED_TESTER_IDS'])
    key = pathlib.Path(os.environ['ASC_KEY_PATH'])
    api = AppleAPI(key, os.environ['ASC_KEY_ID'], os.environ['ASC_ISSUER_ID'])
    private_target(api, app_id, group_id, allowed)
    output = pathlib.Path('testflight-delivery.json')
    report = {'sourceRevision': seal['sourceRevision'], 'buildNumber': number, 'version': seal['version'],
              'ipaSHA256': seal['ipaSHA256'], 'processingVerified': False,
              'groupAssociationVerified': False, 'availableToInternalTesters': False}
    def save(): output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    existing = find_build(api, app_id, number)
    if existing is None:
        report['uploadCommandExitCode'] = upload(ipas[0], key, os.environ['ASC_KEY_ID'], os.environ['ASC_ISSUER_ID'])
        save()  # Even a transport error can mean Apple accepted the upload.
    deadline = time.monotonic() + 1800
    build = wait_valid(api, app_id, number, seal['version'], deadline)
    report.update(buildID=build['id'], processingVerified=True, buildAudienceType='INTERNAL_ONLY'); save()
    # Recheck privacy immediately before the sole access-changing operation.
    private_target(api, app_id, group_id, allowed)
    if build['attributes'].get('usesNonExemptEncryption') is None:
        api.request('builds/' + build['id'], method='PATCH', body={'data': {'type': 'builds', 'id': build['id'],
            'attributes': {'usesNonExemptEncryption': False}}})
    state = attach_and_confirm(api, build, group_id, deadline)
    report.update(groupAssociationVerified=True, availableToInternalTesters=True, internalBuildState=state,
                  appStoreConnectURL='https://appstoreconnect.apple.com/apps/' + app_id + '/testflight/ios'); save()
    print('Apple confirms VALID internal-only build and the configured private group in beta testing.')


if __name__ == '__main__': main()
