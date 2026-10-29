"""Opt-in EAS Apple controls for the existing native app, never a delivery gate.

The first stage compiles all three Flint Apple slices and executes the same
small independent graph/tokenizer/template/API controls as flint.yml. The
compile stage also compiles the existing app and targeted package tests. It
does not replace the full QPack, LES, parity or visual release checks in CI.
Neither stage signs, submits, downloads full model weights, or runs trained
inference. All commands fail closed and their complete logs stay in artifacts.
"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".work/eas-output"
FLAGS = ["-skipPackagePluginValidation", "-skipMacroValidation"]


def plan(stage):
    py = sys.executable
    steps = [
        ("python-tests", [py, "-m", "unittest", "discover", "-s", "scripts/tests", "-p", "test_*.py", "-v"]),
        ("flint-frameworks", [py, "scripts/build_flint_runtime.py"]),
        ("flint-graphs", [py, "scripts/verify_flint_controls.py", "--control", ".work/flint-build-macos/bin/flint-control",
            "--quantizer", ".work/flint-build-macos/bin/flint-quantize-control", "--output", ".work/flint-eas-graph"]),
        ("flint-metadata", [py, "scripts/fetch_flint_metadata.py"]),
        ("flint-protocol", [py, "scripts/verify_flint_protocol.py", "--prefix", ".work/flint-inputs/gguf-prefix.bin",
            "--tokenizer", ".work/flint-inputs/tokenizer.json", "--native-bin", ".work/flint-build-macos/bin",
            "--output", ".work/flint-eas-protocol"]),
    ]
    for name, fixture in [("f32", "normal/synthetic-flint-f32.gguf"), ("q4-1", "normal/q4_1/synthetic-flint-q4_1.gguf"),
                          ("256-experts", "wide-router/synthetic-flint-f32.gguf")]:
        steps.append(("flint-api-" + name, [".work/flint-build-macos/bin/flint-api-control",
            ".work/flint-eas-graph/" + fixture, ".work/flint-eas-api-" + name]))
    if stage == "compile":
        steps += [
            ("tokenizer-frameworks", [py, "scripts/build_tokenizer_xcframework.py"]),
            ("les-frameworks", [py, "scripts/build_ornith_les.py"]),
            ("les-framework-integrity", [py, "scripts/verify_ornith_les_runtime.py"]),
            ("resolve-packages", ["xcodebuild", "-resolvePackageDependencies", "-project", "Apps/Minirun/Minirun.xcodeproj",
                "-scheme", "Minirun-macOS", *FLAGS]),
            ("mac-app-compile", ["xcodebuild", "build", "-project", "Apps/Minirun/Minirun.xcodeproj", "-scheme", "Minirun-macOS",
                "-destination", "platform=macOS", "CODE_SIGNING_ALLOWED=NO", *FLAGS]),
            ("flint-swift-tests", ["xcodebuild", "test", "-scheme", "minirun-Package", "-destination", "platform=macOS",
                "-only-testing:FlintEngineTests", "-only-testing:MinirunKitTests/LocalFlintProtocolTests",
                "-only-testing:MinirunKitTests/LocalModelAPITests", *FLAGS]),
            ("iphone-release-compile", ["xcodebuild", "build", "-project", "Apps/Minirun/Minirun.xcodeproj", "-scheme", "Minirun-iOS",
                "-configuration", "Release", "-destination", "generic/platform=iOS", "CODE_SIGNING_ALLOWED=NO", *FLAGS]),
        ]
    elif stage != "native":
        raise ValueError("Unknown EAS validation stage")
    return steps


def host_check():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("EAS native checks require a real Apple-silicon Mac")
    xcode = subprocess.check_output(["xcodebuild", "-version"], text=True)
    if not xcode.startswith("Xcode 26.2\n"):
        raise RuntimeError("This validation profile requires the pinned Xcode 26.2")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = os.environ.get("EAS_BUILD_GIT_COMMIT_HASH")
    if remote and revision != remote:
        raise RuntimeError("EAS source revision does not match the uploaded Git commit")
    return {"platform": "macOS", "architecture": "arm64", "xcode": xcode.strip(), "sourceRevision": revision}


def execute(stage):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {"schemaVersion": 1, "stage": stage, "status": "running", "steps": [],
        "signingAttempted": False, "testFlightUploaded": False, "trainedInferenceVerified": False,
        "iPhoneVerified": False, "fullReleaseGatesPassed": False}
    path = OUTPUT / "validation.json"

    def save():
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        record["host"] = host_check()
        env = os.environ.copy()
        env.update({"FLASH_LOCAL_ENABLE_LES": "1", "FLASH_LOCAL_ENABLE_FLINT": "1",
                    "ORNITH_LES_CPU_CONTROL": "1", "GITHUB_WORKSPACE": str(ROOT)})
        for name, command in plan(stage):
            print("Starting native check: " + name, flush=True)
            started = time.monotonic()
            with (OUTPUT / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            item = {"name": name, "exitCode": result.returncode, "seconds": time.monotonic() - started}
            record["steps"].append(item)
            save()
            if result.returncode:
                print((OUTPUT / (name + ".log")).read_text(errors="replace")[-16000:], flush=True)
                raise RuntimeError("Native check failed: " + name)
            if name.startswith("flint-api-"):
                text = (OUTPUT / (name + ".log")).read_text()
                if "FLINT_API_CONTROL" not in text or "routingTrace=true" not in text or "unload=true" not in text:
                    raise RuntimeError("Native API success marker missing: " + name)
            print("Passed: " + name, flush=True)
        record["status"] = "passed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        save()
        collect()


def collect():
    """Only known synthetic diagnostics: no recursive workspace or model upload."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    patterns = [".work/eas-output/*.json", ".work/eas-output/*.log",
        ".work/flint-runtime/build.log", ".work/flint-runtime/build-manifest.json",
        ".work/tokenizer-build.log", ".work/runtime/build-manifest.json",
        ".work/ornith-les-runtime/build.log", ".work/ornith-les-runtime/build-manifest.json",
        ".work/flint-eas-graph/**/*.json", ".work/flint-eas-graph/**/*.log",
        ".work/flint-eas-protocol/**/*.json"]
    files = {p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink()}
    index = []
    with zipfile.ZipFile(OUTPUT / "reports.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.stat().st_size > 64 * 1024 * 1024:
                raise RuntimeError("Unexpectedly large diagnostic log")
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, relative)
            index.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        archive.writestr("index.json", json.dumps(index, indent=2) + "\n")
    print("Collected bounded native diagnostics; no full model or signing material", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["native", "compile"], default="native")
    parser.add_argument("--plan", action="store_true", help="Print commands only; does not execute a build")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.plan:
        print(json.dumps(plan(args.stage), indent=2))
    else:
        execute(args.stage)


if __name__ == "__main__":
    main()
