"""Build a symbol-isolated LES XCFramework on the free Apple-silicon CI host."""
import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
from prepare_ornith_les import prepare, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mac-only', action='store_true')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Use the selected Apple-silicon Xcode host')
    root = pathlib.Path(__file__).resolve().parents[1]
    runtime = root / '.work/ornith-les-runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    expected = set((root/'Experiments/OrnithLES/Native/exports.txt').read_text().split())
    records, frameworks = [], []
    with (runtime/'build.log').open('w') as output:
        def run(command):
            subprocess.run([str(x) for x in command], cwd=root, stdout=output, stderr=subprocess.STDOUT, check=True)
        try:
            source = prepare(root, run)
            platforms = [('macos', 'macosx', '14.0')]
            if not args.mac_only:
                platforms += [('ios', 'iphoneos', '18.0'), ('ios-simulator', 'iphonesimulator', '18.0')]
            for name, sdk, minimum in platforms:
                build = root/'.work'/('ornith-les-'+name)
                command = ['cmake', '-S', root/'Experiments/OrnithLES/Native', '-B', build,
                    f'-DOMNI_SOURCE={source}', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_OSX_ARCHITECTURES=arm64', f'-DCMAKE_OSX_SYSROOT={sdk}',
                    f'-DCMAKE_OSX_DEPLOYMENT_TARGET={minimum}', '-DGGML_CPU_ARM_ARCH=armv8-a']
                if name != 'macos':
                    command += ['-G', 'Xcode', '-DCMAKE_SYSTEM_NAME=iOS',
                        '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY',
                        '-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO']
                run(command)
                targets = ['FlashOrnithLES'] + (['ornith-les-control', 'ornith-les-mixed-fixture', 'ornith-les-header-control'] if name == 'macos' else [])
                run(['cmake', '--build', build, '--config', 'Release', '--target', *targets, '--parallel', '3'])
                # Xcode also creates an EagerLinkingTBDs .framework directory
                # containing only a text stub. Select an actual Mach-O product.
                candidates = [p for p in build.rglob('FlashOrnithLES.framework')
                              if (p/'FlashOrnithLES').is_file()]
                if len(candidates) != 1:
                    raise RuntimeError(f'Expected one {name} framework')
                framework = candidates[0]
                modules = framework/'Modules'
                modules.mkdir(exist_ok=True)
                (modules/'module.modulemap').write_text('framework module FlashOrnithLES {\n  umbrella header "FlashOrnithLES.h"\n  export *\n}\n')
                header = framework/'Headers/FlashOrnithLES.h'
                if not header.is_file() or header.read_bytes() != (root/'Experiments/OrnithLES/Native/FlashOrnithLES.h').read_bytes():
                    raise RuntimeError(f'{name} framework is missing its exact public C ABI header')
                probe = runtime/'module-probe.m'
                probe.write_text('@import FlashOrnithLES;\n')
                sdk_path = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-path'], text=True).strip()
                triple = {'macos': 'arm64-apple-macos14.0', 'ios': 'arm64-apple-ios18.0',
                          'ios-simulator': 'arm64-apple-ios18.0-simulator'}[name]
                run(['xcrun', '--sdk', sdk, 'clang', '-target', triple, '-isysroot', sdk_path,
                     '-fmodules', '-fsyntax-only', '-F', framework.parent,
                     '-fmodules-cache-path=' + str(runtime/('module-cache-' + name)), probe])
                binary = framework/'FlashOrnithLES'
                symbols = subprocess.check_output(['xcrun', 'nm', '-gU', binary], text=True)
                actual = {line.split()[-1] for line in symbols.splitlines() if line.strip()}
                if actual != expected:
                    raise RuntimeError(f'LES symbol isolation failed: {actual ^ expected}')
                run(['xcrun', 'lipo', binary, '-verify_arch', 'arm64'])
                dependencies = subprocess.check_output(['otool', '-L', binary], text=True)
                if any(x in dependencies for x in ['libllama.', 'libggml.']):
                    raise RuntimeError('LES framework depends on a second exported llama library')
                frameworks += ['-framework', framework]
                records.append(dict(platform=name, sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    exportedSymbols=sorted(actual), symbolIsolated=True, publicHeaderVerified=True, moduleImportVerified=True))
                print(f'LES framework ready: {name}, isolated exports verified', flush=True)
            if not args.mac_only:
                target = runtime/'FlashOrnithLES.xcframework'
                if target.exists():
                    raise RuntimeError('XCFramework output exists; keep immutable build evidence')
                run(['xcodebuild', '-create-xcframework', *frameworks, '-output', target])
            app_source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            (runtime/'build-manifest.json').write_text(json.dumps(dict(revision=REVISION, appSourceRevision=app_source,
                patchedSource=json.loads((source/'flash-local-source.json').read_text()), slices=records,
                trainedInferenceExecuted=False), indent=2)+'\n')
        except Exception:
            output.flush()
            print((runtime/'build.log').read_text(errors='replace')[-14000:])
            raise
    print(f'Isolated LES framework built: {len(records)} Apple slices; no trained weights')


if __name__ == '__main__':
    main()
