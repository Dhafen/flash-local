"""Export the existing app on EAS after real regression gates; never upload.

The supplied private file contains only the existing distribution certificate,
its password and this app's profile. The Apple API private key stays local.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import secrets
import shutil
import subprocess
import sys
import tempfile
import zipfile

from run_flint_eas import gui_command, host_check
from deliver_testflight import verified_export

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '.work/release-output'
QPACK_INPUTS = ['Experiments/OrnithPerformance', '.github/workflows/ornith-performance.yml',
                'scripts/inspect_ornith_mlx.py', 'scripts/prepare_ornith_mlx_tokenizer.py',
                'scripts/check_qpack_validation_log.py']


def source_fingerprint(root, revision='HEAD'):
    return hashlib.sha256(subprocess.check_output(
        ['git', 'ls-tree', '-r', revision, '--', *QPACK_INPUTS], cwd=root)).hexdigest()


def admission(root):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    record = json.loads((root / '.work/eas-output/validation.json').read_text())
    checks = json.loads((root / '.work/eas-output/regression-checks.json').read_text())
    qpack = json.loads((root / 'Experiments/Flint/evidence/apple-qpack-controls.json').read_text())
    if (record.get('status') != 'passed' or record.get('stage') != 'regression'
            or record.get('host', {}).get('sourceRevision') != revision
            or checks.get('status') != 'passed' or checks.get('sourceRevision') != revision
            or len(checks.get('steps', [])) != 30 or any(x['exitCode'] for x in checks['steps'])
            or qpack.get('status') != 'passed' or len(qpack.get('steps', [])) != 6
            or any(x['exitCode'] for x in qpack['steps'])
            or qpack.get('inputFingerprint') != source_fingerprint(root)):
        raise ValueError('Complete same-source regression and unchanged passed QPack inputs are required before signing')
    return revision


def decode_private_file(path):
    if not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError('Bounded signing input is required')
    data = json.loads(path.read_text())
    if set(data) != {'teamID', 'p12', 'password', 'profile', 'build'}:
        raise ValueError('Only app-scoped manual signing inputs are allowed')
    import re
    if not re.fullmatch(r'[A-Z0-9]{10}', data['teamID']) or not re.fullmatch(r'[1-9][0-9]*', str(data['build'])):
        raise ValueError('Invalid team or build number')
    for key in ('p12', 'profile'):
        data[key] = base64.b64decode(data[key], validate=True)
        if not 0 < len(data[key]) <= 131072: raise ValueError('Unbounded signing resource')
    if not isinstance(data['password'], str) or not 1 <= len(data['password']) <= 256:
        raise ValueError('Missing certificate password')
    return data


def private_error_summary(path, redactions):
    """Expose actionable Xcode errors without exporting the private log."""
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        tail = stream.read(32768).decode(errors='replace')
    for value in redactions:
        if value:
            tail = tail.replace(value, '[withheld]')
    return '\n'.join(line[:1000] for line in tail.splitlines()
                     if 'error:' in line.lower() or 'error exportarchive:' in line.lower())[-6000:]


def package_export(directory, seal_path, output, source, build, run_id):
    """Keep a verified IPA even when Xcode omits optional export metadata."""
    files = list(directory.glob('*.ipa'))
    if len(files) != 1:
        raise ValueError('Exactly one verified exported IPA is required')
    verified_export(files[0], seal_path, source, build, run_id)
    metadata = directory / 'AppStoreInfo.plist'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        # Publish the valuable verified binary first. A later metadata issue
        # must not discard it when the temporary keychain/archive is cleaned.
        archive.write(files[0], 'FlashLocal.ipa')
        archive.write(seal_path, 'ios-signing-probe.json')
        if metadata.is_file():
            if not 0 < metadata.stat().st_size <= 2 * 1024 * 1024:
                raise ValueError('Unexpected optional Xcode metadata size; verified IPA has been preserved')
            if not isinstance(plistlib.loads(metadata.read_bytes()), dict):
                raise ValueError('Invalid optional Xcode metadata; verified IPA has been preserved')
            archive.write(metadata, 'AppStoreInfo.plist')
    return {'appStoreInfoIncluded': metadata.is_file()}


def main():
    host_check()
    revision = admission(ROOT)  # Before reading any certificate/private input.
    env = dict(os.environ, FLASH_LOCAL_ENABLE_LES='1', FLASH_LOCAL_ENABLE_FLINT='1')
    if subprocess.check_output(['launchctl', 'managername'], text=True).strip() != 'Aqua':
        # Only the secret file's path is forwarded, never its contents.
        env['FLINT_SIGNING_INPUT_PATH'] = os.environ['FLINT_SIGNING_INPUT_PATH']
        command = gui_command([sys.executable, str(Path(__file__).resolve())], env, os.getuid())
        subprocess.run(command, check=True, cwd=ROOT)
        return
    data = decode_private_file(Path(os.environ['FLINT_SIGNING_INPUT_PATH']))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    root_temp = Path(tempfile.mkdtemp(prefix='flash-local-eas-signing-'))
    signing = root_temp / 'flash-local-signing'
    signing.mkdir(mode=0o700)
    keychain = root_temp / 'signing.keychain-db'
    installed_profile = None
    installed_profile_owned = False
    original_keychains = subprocess.check_output(['security', 'list-keychains', '-d', 'user'], text=True)
    original_keychains = [line.strip().strip('"') for line in original_keychains.splitlines() if line.strip()]
    report = {'sourceRevision': revision, 'build': str(data['build']), 'signedExportVerified': False,
              'testFlightUploaded': False, 'trainedModelInferenceVerified': False, 'physicalIPhoneVerified': False}
    try:
        p12 = signing / 'distribution.p12'; p12.write_bytes(data['p12']); p12.chmod(0o600)
        profile = signing / 'profile.mobileprovision'; profile.write_bytes(data['profile']); profile.chmod(0o600)
        decoded = plistlib.loads(subprocess.check_output(['security', 'cms', '-D', '-i', str(profile)]))
        import uuid
        identifier = str(uuid.UUID(decoded['UUID']))
        if decoded.get('TeamIdentifier') != [data['teamID']]: raise ValueError('Wrong profile team')
        profile_directory = Path.home() / 'Library/MobileDevice/Provisioning Profiles'
        profile_directory.mkdir(parents=True, exist_ok=True)
        installed_profile = profile_directory / (identifier + '.mobileprovision')
        if installed_profile.exists(): raise ValueError('Refusing to replace an existing installed profile')
        with installed_profile.open('xb') as stream: stream.write(data['profile'])
        installed_profile_owned = True
        password = secrets.token_hex(32)
        with (signing / 'private-signing.log').open('wb') as log:
            def run(command, **kwargs):
                result = subprocess.run([str(x) for x in command], cwd=ROOT,
                                        stdout=log, stderr=subprocess.STDOUT, **kwargs)
                if result.returncode:
                    # CalledProcessError would stringify certificate passwords
                    # from security's argv. Private diagnostics never go to CI.
                    log.flush()
                    summary = private_error_summary(signing / 'private-signing.log',
                        [data['password'], password, base64.b64encode(data['p12']).decode(),
                         base64.b64encode(data['profile']).decode(), data['teamID']])
                    if summary:
                        print(summary, flush=True)
                    raise RuntimeError('Private signing/export step failed: ' + Path(str(command[0])).name)
            run(['security', 'create-keychain', '-p', password, keychain])
            run(['security', 'set-keychain-settings', '-lut', '7200', keychain])
            run(['security', 'unlock-keychain', '-p', password, keychain])
            run(['security', 'import', p12, '-P', data['password'], '-T', '/usr/bin/codesign', '-t', 'cert', '-f', 'pkcs12', '-k', keychain])
            run(['security', 'set-key-partition-list', '-S', 'apple-tool:,apple:,codesign:', '-k', password, keychain])
            run(['security', 'list-keychains', '-d', 'user', '-s', keychain, *original_keychains])
            archive = root_temp / 'FlashLocal.xcarchive'
            run(['xcodebuild', 'archive', '-project', 'Apps/Minirun/Minirun.xcodeproj', '-scheme', 'Minirun-iOS',
                 '-configuration', 'Release', '-destination', 'generic/platform=iOS', '-skipPackagePluginValidation',
                 '-skipMacroValidation', '-archivePath', archive, 'CODE_SIGNING_ALLOWED=NO',
                 'CURRENT_PROJECT_VERSION=' + str(data['build'])], env=env)
            run([sys.executable, 'scripts/check_ios_archive.py', archive, '--output', OUTPUT / 'archive-check.json'])
            export_env = dict(env, RUNNER_TEMP=str(root_temp), APPLE_TEAM_ID=data['teamID'], GITHUB_SHA=revision,
                              GITHUB_RUN_ID=os.environ['EAS_BUILD_ID'])
            run([sys.executable, 'scripts/probe_automatic_export.py', archive, '--profile', profile], env=export_env)
        seal = ROOT / 'ios-signing-probe.json'
        report.update(package_export(signing / 'export', seal, OUTPUT / 'release.zip',
            revision, str(data['build']), os.environ['EAS_BUILD_ID']))
        report['signedExportVerified'] = True
        print('Verified signed internal-only export prepared; no API private key copied and no TestFlight upload attempted')
    finally:
        (OUTPUT / 'release-status.json').write_text(json.dumps(report, indent=2) + '\n')
        subprocess.run(['security', 'list-keychains', '-d', 'user', '-s', *original_keychains], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['security', 'delete-keychain', str(keychain)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if installed_profile_owned and installed_profile and installed_profile.exists(): installed_profile.unlink()
        if root_temp.resolve().parent != Path(tempfile.gettempdir()).resolve(): raise RuntimeError('Unexpected signing cleanup root')
        shutil.rmtree(root_temp)


def collect():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public_files = [(ROOT / '.work/eas-output/reports.zip', 'validation-reports.zip'),
                    (OUTPUT / 'release.zip', 'release.zip'), (OUTPUT / 'release-status.json', 'release-status.json'),
                    (OUTPUT / 'archive-check.json', 'archive-check.json'),
                    (ROOT / 'ios-signing-probe.json', 'ios-signing-probe.json')]
    with zipfile.ZipFile(OUTPUT / 'result.zip', 'w', compression=zipfile.ZIP_STORED) as out:
        for path, name in public_files:
            if path.is_file():
                if path.stat().st_size > 600_000_000: raise ValueError('Unbounded public export artifact')
                out.write(path, name)


if __name__ == '__main__':
    if sys.argv[1:] == ['--collect']: collect()
    else: main()
