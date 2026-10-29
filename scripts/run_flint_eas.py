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
import plistlib
import re
import subprocess
import sys
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".work/eas-output"
FLAGS = ["-skipPackagePluginValidation", "-skipMacroValidation"]


def git_symlink_plan(root):
    """Validate the Git-declared links before repairing a Windows-made archive."""
    root = root.resolve()
    entries = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"])
    links = []
    for entry in entries.split(b"\0"):
        if not entry.startswith(b"120000 "):
            continue
        header, raw_name = entry.split(b"\t", 1)
        if header.split()[2] != b"0":
            raise RuntimeError("Unmerged symbolic link in build source")
        name = raw_name.decode("utf-8")
        path = root / name
        if not path.parent.resolve().is_relative_to(root):
            raise RuntimeError("Build link parent escapes source")
        target_bytes = subprocess.check_output(["git", "-C", str(root), "show", ":" + name])
        target = target_bytes.decode("utf-8")
        if not target or "\0" in target or "\\" in target or pathlib.PurePosixPath(target).is_absolute():
            raise RuntimeError("Build link target must be a relative source path")
        destination = (path.parent / target).resolve()
        if not destination.is_relative_to(root) or not destination.is_file():
            raise RuntimeError("Build link must reference an existing file inside source")
        if path.is_symlink():
            if os.readlink(path) != target:
                raise RuntimeError("Existing build symbolic link was modified")
        elif not path.is_file() or path.read_bytes() != target_bytes:
            raise RuntimeError("Refusing to replace modified build link content")
        links.append((name, target))
    return links


def restore_git_symlinks(root):
    links = git_symlink_plan(root)
    if links:
        # Only this ephemeral Apple checkout changes. Windows and the original
        # app checkout keep their existing Git configuration and file contents.
        subprocess.run(["git", "-C", str(root), "config", "core.symlinks", "true"], check=True)
        for name, target in links:
            subprocess.run(["git", "-C", str(root), "checkout-index", "--force", "--", name], check=True)
            path = root / name
            if not path.is_symlink() or os.readlink(path) != target:
                raise RuntimeError("Git source link restoration failed")
    return [name for name, _ in links]


def plan(stage):
    py = sys.executable
    if stage == "qpack":
        return [("python-tests", [py, "-m", "unittest", "discover", "-s", "scripts/tests", "-p", "test_*.py", "-v"]),
                ("apple-test-host", [py, "scripts/run_flint_eas.py", "--prepare-tests"]),
                ("qpack-ci", [py, "scripts/run_eas_regressions.py", "qpack"])]
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
    if stage in ("compile", "visual"):
        steps += [
            ("tokenizer-frameworks", [py, "scripts/build_tokenizer_xcframework.py"]),
            ("les-frameworks", [py, "scripts/build_ornith_les.py"]),
            ("les-framework-integrity", [py, "scripts/verify_ornith_les_runtime.py"]),
            ("resolve-packages", ["xcodebuild", "-resolvePackageDependencies", "-project", "Apps/Minirun/Minirun.xcodeproj",
                "-scheme", "Minirun-macOS", *FLAGS]),
            ("mac-app-compile", ["xcodebuild", "build", "-project", "Apps/Minirun/Minirun.xcodeproj", "-scheme", "Minirun-macOS",
                "-destination", "platform=macOS", "CODE_SIGNING_ALLOWED=NO", *FLAGS]),
            ("iphone-release-compile", ["xcodebuild", "build", "-project", "Apps/Minirun/Minirun.xcodeproj", "-scheme", "Minirun-iOS",
                "-configuration", "Release", "-destination", "generic/platform=iOS", "CODE_SIGNING_ALLOWED=NO", *FLAGS]),
            ("apple-test-host", [py, "scripts/run_flint_eas.py", "--prepare-tests"]),
        ]
    elif stage == "regression":
        steps += [("apple-test-host", [py, "scripts/run_flint_eas.py", "--prepare-tests"]),
                  ("regression-ci", [py, "scripts/run_eas_regressions.py", "regression"])]
    elif stage != "native":
        raise ValueError("Unknown EAS validation stage")
    if stage == "visual":
        steps += [
            ("ios-visual-tests", [py, "scripts/run_ios_brand_checks.py", "--timeout-seconds", "900", "--attempts", "1"]),
        ]
    if stage in ("compile", "visual"):
        steps += [
            ("flint-swift-tests", ["xcodebuild", "test", "-scheme", "minirun-Package", "-destination", "platform=macOS",
                "-parallel-testing-enabled", "NO", "-only-testing:FlintEngineTests",
                "-only-testing:MinirunKitTests/LocalFlintProtocolTests", "-only-testing:MinirunKitTests/LocalModelAPITests", *FLAGS]),
        ]
    if stage == "visual":
        steps += [
            ("mac-visual-tests", ["xcodebuild", "test", "-project", "Apps/Minirun/Minirun.xcodeproj", "-scheme", "Minirun-macOS",
                "-destination", "platform=macOS", "-parallel-testing-enabled", "NO", "CODE_SIGNING_ALLOWED=NO", *FLAGS,
                "-only-testing:MinirunTests/ForkBrandRenderTests", "-only-testing:MinirunTests/REAMChatRenderTests",
                "-only-testing:MinirunTests/OrnithPerformanceRenderTests", "-only-testing:MinirunTests/OrnithLESRenderTests",
                "-only-testing:MinirunTests/FlintRenderTests", "-resultBundlePath", ".work/eas-visual/mac.xcresult"]),
            ("mac-visual-export", ["xcrun", "xcresulttool", "export", "attachments", "--path", ".work/eas-visual/mac.xcresult",
                "--output-path", ".work/eas-visual/mac"]),
        ]
    return steps


def prepare_apple_test_host():
    """Initialize Apple's installed test components on the ephemeral EAS Mac.

    EAS can compile binaries before testmanagerd has been registered. Use the
    ordinary Xcode first-launch setup, without changing security permissions,
    loading hand-written daemons or treating setup as an executed test.
    """
    if platform.system() != "Darwin" or not os.environ.get("EAS_BUILD_ID"):
        raise RuntimeError("Test-host initialization is only for the ephemeral EAS Mac")
    for command in (["launchctl", "managername"], ["xcodebuild", "-checkFirstLaunchStatus"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        print(json.dumps({"command": command, "exitCode": result.returncode,
                          "output": (result.stdout + result.stderr)[-4000:]}), flush=True)
    # A missing launchd registration can remain even when the status check
    # says the installed components are current. Re-run Apple's own setup.
    subprocess.run(["sudo", "-n", "xcodebuild", "-runFirstLaunch"], check=True, timeout=300)
    for domain in ("user/" + str(os.getuid()), "gui/" + str(os.getuid()), "system"):
        for label in ("com.apple.dt.testmanagerd", "com.apple.testmanagerd"):
            result = subprocess.run(["launchctl", "print", domain + "/" + label],
                                    capture_output=True, text=True, timeout=30)
            # Do not export launchd environments. Only presence/exit status is
            # needed to diagnose whether the worker can launch Apple tests.
            print(json.dumps({"service": domain + "/" + label, "exitCode": result.returncode}), flush=True)
    for directory in ("/Library/LaunchAgents", "/System/Library/LaunchAgents", "/Library/LaunchDaemons"):
        for path in sorted(pathlib.Path(directory).glob("*testmanager*.plist")):
            with path.open("rb") as stream:
                config = plistlib.load(stream)
            print(json.dumps({"installedAgent": str(path), "label": config.get("Label"),
                              "machServices": list(config.get("MachServices", {})),
                              "sessionTypes": config.get("LimitLoadToSessionType")}), flush=True)
    print("Apple host preparation completed; no test result is implied", flush=True)


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


def verify_step_log(name, content):
    """A zero process status must not admit empty or skipped test selections."""
    if name.startswith("flint-api-"):
        if "FLINT_API_CONTROL" not in content or "routingTrace=true" not in content or "unload=true" not in content:
            raise RuntimeError("Native API success marker missing: " + name)
    suites = ()
    if name == "flint-swift-tests":
        suites = ("FlintPolicyTests", "FlintStreamTests", "LocalFlintProtocolTests", "LocalModelAPITests")
        for marker in ("ORNITH_LOCAL_API realTCP=true auth=true json=true sse=true reasoningSeparated=true",
                       "LOCAL_API_TOOL_ROUNDTRIP realTCP=true structuredArguments=true matchedToolResult=true"):
            if marker not in content:
                raise RuntimeError("Required API round trip did not execute")
    elif name == "mac-visual-tests":
        suites = ("ForkBrandRenderTests", "REAMChatRenderTests", "OrnithPerformanceRenderTests",
                  "OrnithLESRenderTests", "FlintRenderTests")
        if "FLINT_RENDER unloadedPanes=3 appearances=2 realControls=true modelLoaded=false trainedInference=false" not in content:
            raise RuntimeError("Flint native captures were not executed")
    for suite in suites:
        if ("Test Suite '" + suite + "' passed") not in content or not re.search(
                r"Test Case [^\n]*\b" + re.escape(suite) + r"\b[^\n]* passed", content):
            raise RuntimeError("Required XCTest suite did not execute and pass: " + suite)


def gui_command(command, env, uid):
    """Join the already logged-in EAS user's bootstrap, then drop root again.

    The worker launches custom steps in the System bootstrap. Apple's genuine
    testmanagerd is in gui/<uid>; no agent is installed or reconfigured here.
    Only explicit build settings are carried across sudo, never credentials.
    """
    if type(uid) is not int or uid <= 0:
        raise RuntimeError("Apple tests must run as the existing non-root EAS user")
    settings = {key: env[key] for key in (
        "PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "DEVELOPER_DIR",
        "GITHUB_WORKSPACE", "FLASH_LOCAL_ENABLE_LES", "FLASH_LOCAL_ENABLE_FLINT",
        "ORNITH_LES_CPU_CONTROL", "TEST_RUNNER_ORNITH_LES_CPU_CONTROL",
        "EAS_BUILD_ID", "EAS_BUILD_GIT_COMMIT_HASH",
        "FLINT_SIGNING_INPUT_PATH",
    ) if key in env}
    return ["sudo", "-n", "/bin/launchctl", "asuser", str(uid),
            "sudo", "-n", "-H", "-u", "#" + str(uid), "--", "/usr/bin/env", "-i",
            *[key + "=" + value for key, value in settings.items()], *command]


def check_gui_context(env, uid):
    probe = subprocess.run(gui_command(["/bin/launchctl", "managername"], env, uid),
                           capture_output=True, text=True, timeout=30)
    if probe.returncode or probe.stdout.strip() != "Aqua":
        raise RuntimeError("Apple XCTest did not enter the existing Aqua session: " +
                           (probe.stdout + probe.stderr)[-2000:])
    return {"manager": "Aqua", "userID": uid, "credentialsInherited": False}


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
        record["restoredSymbolicLinks"] = restore_git_symlinks(ROOT)
        if stage == "visual":
            (ROOT / ".work/eas-visual").mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"FLASH_LOCAL_ENABLE_LES": "1", "FLASH_LOCAL_ENABLE_FLINT": "1",
                    "ORNITH_LES_CPU_CONTROL": "1", "TEST_RUNNER_ORNITH_LES_CPU_CONTROL": "1",
                    "GITHUB_WORKSPACE": str(ROOT)})
        if stage != "native":
            # Verify the host transition before spending time rebuilding native
            # slices. A missing desktop session must remain a visible blocker.
            record["testBootstrap"] = check_gui_context(env, os.getuid())
            save()
            print("Verified existing Aqua test session; build user preserved", flush=True)
        for name, command in plan(stage):
            print("Starting native check: " + name, flush=True)
            started = time.monotonic()
            with (OUTPUT / (name + ".log")).open("w", encoding="utf-8") as log:
                if name in ("flint-swift-tests", "mac-visual-tests", "qpack-ci", "regression-ci"):
                    context = check_gui_context(env, os.getuid())
                    log.write(json.dumps({"testBootstrap": context}) + "\n")
                    log.write("EAS_TEST_BOOTSTRAP manager=Aqua sameUser=true credentialsInherited=false\n")
                    log.flush()
                    command = gui_command(command, env, os.getuid())
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            item = {"name": name, "exitCode": result.returncode, "seconds": time.monotonic() - started}
            record["steps"].append(item)
            save()
            if result.returncode:
                print((OUTPUT / (name + ".log")).read_text(errors="replace")[-16000:], flush=True)
                raise RuntimeError("Native check failed: " + name)
            verify_step_log(name, (OUTPUT / (name + ".log")).read_text(errors="replace"))
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
        ".work/flint-eas-protocol/**/*.json", ".work/eas-visual/mac/**/*.png",
        ".work/eas-visual/mac/**/*.json", "brand-ios/**/*.png", "brand-ios/**/*.json",
        "brand-macos/**/*.png", "brand-macos/**/*.json", "streaming-tests.log", "parity-tests.log",
        "app-tests.log", "package-tests.log", "flint-swift-tests.log", "ios-build.log", "flint-device-build.log", "mac-brand.log",
        ".work/les-*.log", ".work/les-*.json", ".work/ornith-thinking-tokenizer.log",
        ".work/swiftlet-validation/*.log", "Experiments/OrnithPerformance/*.log", "ios-brand*.log"]
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
    parser.add_argument("--stage", choices=["native", "compile", "visual", "qpack", "regression"], default="native")
    parser.add_argument("--plan", action="store_true", help="Print commands only; does not execute a build")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--prepare-tests", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.prepare_tests:
        prepare_apple_test_host()
    elif args.plan:
        print(json.dumps(plan(args.stage), indent=2))
    else:
        execute(args.stage)


if __name__ == "__main__":
    main()
