"""Build canonical tokenizer/Jinja and diagnostic CPU API for Apple arm64.

Run before Swift package resolution. Nothing downloaded is a model; the local
XCFramework is ignored by Git and contains CPU code plus a narrow C header.
"""
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess

from llama_reference_source import REVISION, prepare


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Tokenizer runtime requires the selected Apple-silicon Xcode host")
    root = pathlib.Path(__file__).resolve().parents[1]
    work = root / ".work"
    work.mkdir(exist_ok=True)
    runtime = work / "runtime"
    runtime.mkdir(exist_ok=True)
    log = work / "tokenizer-build.log"
    records = []
    with log.open("w", encoding="utf-8") as output:
        def run(args):
            subprocess.run([str(a) for a in args], cwd=root, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        try:
            source = prepare(root, run)
            headers = runtime / "headers"
            headers.mkdir(exist_ok=True)
            shutil.copyfile(root / "Sources/CLlamaTokenizer/include/FlashLlamaTokenizer.h",
                            headers / "FlashLlamaTokenizer.h")
            shutil.copyfile(root / "Sources/CLlamaTokenizer/include/FlashLlamaReference.h",
                            headers / "FlashLlamaReference.h")
            (headers / "module.modulemap").write_text(
                'module LlamaTokenizerRuntime {\n  header "FlashLlamaTokenizer.h"\n  export *\n}\n', encoding="utf-8")
            libraries = []
            for name, sdk, minimum in [("macos", "macosx", "14.0"),
                                       ("ios", "iphoneos", "18.0"),
                                       ("ios-simulator", "iphonesimulator", "18.0")]:
                build = work / ("reference-build" if name == "macos" else "tokenizer-" + name)
                args = ["cmake", "-S", root / "Tests/ReferenceHarness", "-B", build,
                        f"-DLLAMA_SOURCE={source}", "-DCMAKE_BUILD_TYPE=Release",
                        "-DCMAKE_OSX_ARCHITECTURES=arm64", f"-DCMAKE_OSX_SYSROOT={sdk}",
                        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={minimum}",
                        "-DGGML_NATIVE=OFF", "-DGGML_CPU_ARM_ARCH=armv8-a", "-DGGML_ACCELERATE=OFF",
                        "-DGGML_CPU_KLEIDIAI=OFF", "-DGGML_LLAMAFILE=OFF"]
                if name != "macos":
                    args += ["-G", "Xcode", "-DCMAKE_SYSTEM_NAME=iOS",
                             "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
                             "-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO",
                             "-DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO"]
                run(args)
                run(["cmake", "--build", build, "--config", "Release", "--target",
                     "flash-local-tokenizer-runtime", "--parallel", "3"])
                names = ["libflash-local-tokenizer-runtime.a", "libllama.a", "libggml.a",
                         "libggml-base.a", "libggml-cpu.a"]
                archives = []
                for archive_name in names:
                    matches = list(build.rglob(archive_name))
                    if len(matches) != 1:
                        raise RuntimeError(f"Expected exactly one {name} {archive_name}, found {len(matches)}")
                    archives.append(matches[0])
                directory = runtime / name
                directory.mkdir(exist_ok=True)
                combined = directory / "libLlamaTokenizerRuntime.a"
                run(["xcrun", "libtool", "-static", "-o", combined, *archives])
                run(["xcrun", "lipo", combined, "-verify_arch", "arm64"])
                libraries += ["-library", str(combined), "-headers", str(headers)]
                records.append({"platform": name, "sdk": sdk, "minimumOS": minimum,
                                "architecture": "arm64", "bytes": combined.stat().st_size,
                                "sha256": hashlib.sha256(combined.read_bytes()).hexdigest()})
            target = runtime / "LlamaTokenizerRuntime.xcframework"
            if target.exists():
                # Only this exact generated artifact under .work/runtime may be replaced.
                if target.resolve().parent != runtime.resolve() or target.is_symlink():
                    raise RuntimeError("Unexpected runtime output location")
                shutil.rmtree(target)
            run(["xcodebuild", "-create-xcframework", *libraries, "-output", target])
        except Exception:
            output.flush()
            print(log.read_text(encoding="utf-8", errors="replace")[-18000:])
            raise
    (runtime / "build-manifest.json").write_text(json.dumps({
        "llamaRevision": REVISION, "slices": records,
        "xcode": subprocess.check_output(["xcodebuild", "-version"], text=True).strip(),
        "tokenizerPath": {"vocabularyOnly": True, "inferenceContextCreated": False},
        "diagnosticCPUReference": {"included": True, "maximumContextTokens": 128,
                                   "loadMode": "mmap", "lazyMode": "on", "gpuLayers": 0,
                                   "weightRepacking": False, "requiresMemoryGovernor": True},
        "trainedInferenceExecutedDuringBuild": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Canonical vocabulary-only runtime built for {len(records)} Apple platforms at {REVISION}")


if __name__ == "__main__":
    main()
