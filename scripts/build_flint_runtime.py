"""Build the isolated Apple Flint XCFramework; no signing/workflow changes."""
import argparse
import hashlib
import json
import pathlib
import platform
import plistlib
import subprocess
from prepare_flint_runtime import prepare, ROOT, NATIVE


def verify_existing(source, output, native=NATIVE, mac_only=False):
    """Verify a prior immutable build; never replace or silently rebuild it."""
    record = json.loads((output / "build-manifest.json").read_text())
    source_record = json.loads((source / "flash-local-flint-source.json").read_text())
    if record["source"] != source_record or record.get("trainedInferenceVerified") is not False or record.get("iPhoneVerified") is not False:
        raise ValueError("Existing Flint framework source or validation declaration changed")
    actual_inputs = {str(p.relative_to(native)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in native.iterdir() if p.is_file()}
    if record["nativeFiles"] != actual_inputs:
        raise ValueError("Existing Flint framework native inputs changed")
    exports = set((native / "exports.txt").read_text().split())
    expected = {entry["platform"]: entry for entry in record["slices"]}
    platforms = {"macos"} if mac_only else {"macos", "ios", "ios-simulator"}
    if len(expected) != len(record["slices"]) or set(expected) != platforms or any(
            x.get("moduleImportVerified") is not True or x.get("symbolIsolated") is not True
            or set(x["exportedSymbols"]) != exports for x in expected.values()):
        raise ValueError("Incomplete Flint platform, symbol or module proof")
    framework = (output / "FlashFlint.xcframework").resolve()
    info = plistlib.loads((framework / "Info.plist").read_bytes())
    seen = set()
    for entry in info["AvailableLibraries"]:
        name = entry["SupportedPlatform"]
        if entry.get("SupportedPlatformVariant") == "simulator": name += "-simulator"
        if name not in expected or name in seen or entry["SupportedArchitectures"] != ["arm64"]:
            raise ValueError("Unexpected Flint XCFramework slice")
        seen.add(name)
        library = (framework / entry["LibraryIdentifier"] / entry["LibraryPath"]).resolve()
        if not library.is_relative_to(framework): raise ValueError("Flint framework path escaped output")
        digest = hashlib.sha256()
        with (library / "FlashFlint").open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b""): digest.update(chunk)
        if digest.hexdigest() != expected[name]["sha256"]:
            raise ValueError("Existing Flint framework binary changed")
        if (library / "Headers/FlashFlint.h").read_bytes() != (native / "FlashFlint.h").read_bytes():
            raise ValueError("Existing Flint framework C ABI header changed")
        module = 'framework module FlashFlint {\n  umbrella header "FlashFlint.h"\n  export *\n}\n'
        if (library / "Modules/module.modulemap").read_text() != module:
            raise ValueError("Existing Flint framework module declaration changed")
    if seen != platforms: raise ValueError("Missing Flint XCFramework slice")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true", help="Verify and reuse immutable output; never rebuild")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Apple framework verification requires an Apple-silicon Mac with Xcode. Linux CPU controls are separate.")
    source = prepare()
    output = ROOT / ".work/flint-runtime"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "FlashFlint.xcframework"
    if args.verify_existing:
        count = verify_existing(source, output, mac_only=args.mac_only)
        print(f"FLINT_RUNTIME_REUSE sourceHashes=true binaryHashes=true platforms={count} unchanged=true")
        return
    if target.exists(): raise RuntimeError("Keep immutable XCFramework evidence; use a fresh build directory/worktree")
    platforms = [("macos", "macosx", "14.0")]
    if not args.mac_only: platforms += [("ios", "iphoneos", "18.0"), ("ios-simulator", "iphonesimulator", "18.0")]
    expected = set((NATIVE / "exports.txt").read_text().split())
    frameworks, records = [], []
    with (output / "build.log").open("w", encoding="utf-8") as log:
        def run(command):
            subprocess.run([str(x) for x in command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        try:
            for name, sdk, minimum in platforms:
                build = ROOT / ".work" / ("flint-build-" + name)
                command = ["cmake", "-S", NATIVE, "-B", build, f"-DOMNI_SOURCE={source}", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_OSX_ARCHITECTURES=arm64", f"-DCMAKE_OSX_SYSROOT={sdk}", f"-DCMAKE_OSX_DEPLOYMENT_TARGET={minimum}",
                    "-DGGML_CPU_ARM_ARCH=armv8-a"]
                if name != "macos": command += ["-G", "Xcode", "-DCMAKE_SYSTEM_NAME=iOS",
                    "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY", "-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO"]
                run(command)
                targets = ["FlashFlint"] + (["flint-control", "flint-quantize-control", "flint-api-control", "flint-vocab-control", "flint-prompt-control", "flint-weight-control"] if name == "macos" else [])
                run(["cmake", "--build", build, "--config", "Release", "--target", *targets, "--parallel", "3"])
                products = [p for p in build.rglob("FlashFlint.framework") if (p / "FlashFlint").is_file()]
                if len(products) != 1: raise RuntimeError(f"Expected one actual {name} framework")
                framework = products[0]
                (framework / "Modules").mkdir(exist_ok=True)
                (framework / "Modules/module.modulemap").write_text('framework module FlashFlint {\n  umbrella header "FlashFlint.h"\n  export *\n}\n')
                if (framework / "Headers/FlashFlint.h").read_bytes() != (NATIVE / "FlashFlint.h").read_bytes():
                    raise RuntimeError("Framework C ABI header mismatch")
                symbols = subprocess.check_output(["xcrun", "nm", "-gU", framework / "FlashFlint"], text=True)
                actual = {line.split()[-1] for line in symbols.splitlines() if line.strip()}
                if actual != expected: raise RuntimeError(f"Flint symbol isolation failed: {actual ^ expected}")
                run(["xcrun", "lipo", framework / "FlashFlint", "-verify_arch", "arm64"])
                dependencies = subprocess.check_output(["otool", "-L", framework / "FlashFlint"], text=True)
                if any(x in dependencies for x in ["libllama.", "libggml."]): raise RuntimeError("Unisolated dynamic llama/ggml dependency")
                probe = output / "module-probe.m"
                probe.write_text("@import FlashFlint;\n")
                sdk_path = subprocess.check_output(["xcrun", "--sdk", sdk, "--show-sdk-path"], text=True).strip()
                triple = {"macos": "arm64-apple-macos14.0", "ios": "arm64-apple-ios18.0", "ios-simulator": "arm64-apple-ios18.0-simulator"}[name]
                run(["xcrun", "--sdk", sdk, "clang", "-target", triple, "-isysroot", sdk_path, "-fmodules", "-fsyntax-only",
                    "-F", framework.parent, "-fmodules-cache-path=" + str(output / ("module-cache-" + name)), probe])
                frameworks += ["-framework", framework]
                records.append({"platform": name, "sha256": hashlib.sha256((framework / "FlashFlint").read_bytes()).hexdigest(),
                    "exportedSymbols": sorted(actual), "moduleImportVerified": True, "symbolIsolated": True})
            run(["xcodebuild", "-create-xcframework", *frameworks, "-output", target])
        except Exception:
            log.flush(); print((output / "build.log").read_text(errors="replace")[-12000:]); raise
    manifest = {"schemaVersion": 1, "source": json.loads((source / "flash-local-flint-source.json").read_text()),
        "nativeFiles": {str(p.relative_to(NATIVE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in NATIVE.iterdir() if p.is_file()},
        "slices": records, "trainedInferenceVerified": False, "iPhoneVerified": False}
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Flint XCFramework built with {len(records)} verified slices; no trained model or iPhone benchmark")


if __name__ == "__main__": main()
