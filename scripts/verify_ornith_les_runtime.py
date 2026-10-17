"""Bind the consumed binary to the native/reference validation job's source."""
import hashlib
import json
import pathlib
import plistlib
import subprocess


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    runtime = root/'.work/ornith-les-runtime'
    record = json.loads((runtime/'build-manifest.json').read_text())
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if record['appSourceRevision'] != source or record['trainedInferenceExecuted'] is not False:
        raise ValueError('LES binary is from another source or lacks the synthetic-only declaration')
    expected = {entry['platform']: entry for entry in record['slices']}
    if set(expected) != {'macos', 'ios', 'ios-simulator'} or not all(
            x['symbolIsolated'] and x['publicHeaderVerified'] and x['moduleImportVerified'] for x in expected.values()):
        raise ValueError('Incomplete symbol-isolated platform validation')
    framework = runtime/'FlashOrnithLES.xcframework'
    info = plistlib.loads((framework/'Info.plist').read_bytes())
    seen = set()
    header = (root/'Experiments/OrnithLES/Native/FlashOrnithLES.h').read_bytes()
    for entry in info['AvailableLibraries']:
        platform = entry['SupportedPlatform']
        if entry.get('SupportedPlatformVariant') == 'simulator': platform += '-simulator'
        if platform not in expected or platform in seen or entry['SupportedArchitectures'] != ['arm64']:
            raise ValueError('Unexpected LES XCFramework slice')
        seen.add(platform)
        library = framework/entry['LibraryIdentifier']/entry['LibraryPath']
        if hashlib.sha256((library/'FlashOrnithLES').read_bytes()).hexdigest() != expected[platform]['sha256']:
            raise ValueError('LES executable hash changed during transfer')
        if (library/'Headers/FlashOrnithLES.h').read_bytes() != header:
            raise ValueError('LES C ABI header does not match this source')
    if seen != set(expected): raise ValueError('Missing LES executable slice')
    print('LES runtime source, executable hashes, headers and all three isolated slices verified')


if __name__ == '__main__':
    main()
