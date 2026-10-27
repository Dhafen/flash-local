"""Build the isolated Apple Flint XCFramework; no signing/workflow changes."""
import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
from prepare_flint_runtime import prepare, ROOT, NATIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Apple framework verification requires an Apple-silicon Mac with Xcode. Linux CPU controls are separate.")
    source = prepare()
    output = ROOT / ".work/flint-runtime"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "FlashFlint.xcframework"
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
                targets = ["FlashFlint"] + (["flint-control", "flint-quantize-control", "flint-api-control", "flint-vocab-control", "flint-prompt-control"] if name == "macos" else [])
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
