"""Execute the existing, unsigned CI checks on EAS without weakening assertions.

The tested shell bodies come from the checked-in workflows. This adapter only
supplies EAS's already selected Xcode, the existing Aqua session, and bounded
logs. It never selects an archive, signing, credential or upload step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".work/eas-output"
GROUPS = {
    "qpack": [("ornith-performance.yml", "isolated-engine",
               "Pinned Xcode and isolated Swiftlet source", "Compile experiment for iOS Simulator")],
    "regression": [
        ("ornith-les.yml", "native-controls", "Build isolated Mac runtime for fast numerical controls",
         "Swift memory, comparison and aggregate export controls"),
        ("ci.yml", "build-and-test", "Delivery logic tests",
         "Compile the opt-in Flint app for physical iPhone (unsigned)"),
    ],
}


def resolve_environment(values, root):
    if not isinstance(values, dict):
        raise ValueError("Workflow environment must be an object")
    result = {}
    for key, value in values.items():
        if not isinstance(value, str):
            raise ValueError("Only explicit string build settings are supported")
        value = value.replace("${{ github.workspace }}", str(root))
        # These two opt-in engines are explicitly enabled by the EAS validation
        # profile. Resolve only the exact reviewed GitHub input expressions.
        switches = {
            "FLASH_LOCAL_ENABLE_LES": "${{ inputs.ornith_les && '1' || '0' }}",
            "FLASH_LOCAL_ENABLE_FLINT": "${{ inputs.flint && '1' || '0' }}",
            "ORNITH_LES_CPU_CONTROL": "${{ inputs.ornith_les && '1' || '0' }}",
        }
        if key in switches and value == switches[key]:
            value = "1"
        if ("${{" in value or key.upper() == "TOKEN" or key.upper().endswith("_TOKEN")
                or any(x in key.upper() for x in ("SECRET", "PASSWORD", "APP_STORE"))):
            raise ValueError("Credentials or unresolved workflow expressions are not permitted")
        result[key] = value
    return result


def workflow_steps(filename, job, first, last, root=ROOT):
    # Only EAS orchestration requires this pinned development dependency.
    # Importing the module must not change default GitHub/Python test setup.
    import yaml
    path = root / ".github/workflows" / filename
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = data["jobs"][job]["steps"]
    names = [step.get("name") for step in steps]
    if names.count(first) != 1 or names.count(last) != 1 or names.index(first) > names.index(last):
        raise ValueError("Required CI range is missing or ambiguous")
    defaults = resolve_environment(data.get("env", {}), root)
    defaults.update(resolve_environment(data["jobs"][job].get("env", {}), root))
    result = []
    for step in steps[names.index(first):names.index(last) + 1]:
        if "run" not in step:
            # GitHub artifact actions are replaced by the EAS bounded collector;
            # every shell test/assertion inside this range still runs.
            if step.get("uses") != "actions/upload-artifact@v4":
                raise ValueError("Unexpected action inside the required test range")
            continue
        body = step["run"]
        if step.get("name") == "Pinned Xcode and isolated Swiftlet source":
            original = "sudo xcode-select -s /Applications/Xcode_26.2.app\n"
            if body.count(original) != 1:
                raise ValueError("The pinned host setup changed; review the EAS adapter")
            # EAS uses /Applications/Xcode.app; the actual version is checked
            # by both the parent and this runner before any test executes.
            body = body.replace(original, "")
        if step.get("name") == "Build isolated Flint runtime (Mac, iPhone, Simulator)":
            if body not in ("python3 scripts/build_flint_runtime.py", "python3 scripts/build_flint_runtime.py\n"):
                raise ValueError("Flint build step changed; review immutable reuse")
            # The same EAS job has already built and numerically checked this
            # artifact. Keep its evidence immutable and verify every slice.
            body = "python3 scripts/build_flint_runtime.py --verify-existing\n"
        if any(word in body for word in ("${{", "deliver_testflight.py", "prepare_flash_signing.sh",
                                         "-allowProvisioningUpdates", "xcodebuild archive")):
            raise ValueError("Only fully resolved, unsigned CI test bodies are allowed")
        settings = defaults | resolve_environment(step.get("env", {}), root)
        directory = Path(step.get("working-directory", "."))
        if directory.is_absolute() or not (root / directory).resolve().is_relative_to(root.resolve()):
            raise ValueError("Workflow directory escapes source")
        result.append({"name": step["name"], "workflow": filename, "job": job, "body": body,
                       "bodySHA256": hashlib.sha256(body.encode()).hexdigest(),
                       "directory": directory.as_posix(), "environment": settings})
    if not result:
        raise ValueError("Required test range is empty")
    return result


def plan(group, root=ROOT):
    if group not in GROUPS:
        raise ValueError("Unknown regression group")
    return [step for args in GROUPS[group] for step in workflow_steps(*args, root=root)]


def execute(group):
    if platform.system() != "Darwin" or platform.machine() != "arm64" or not os.environ.get("EAS_BUILD_ID"):
        raise RuntimeError("Regression execution requires the actual EAS Apple host")
    if subprocess.check_output(["launchctl", "managername"], text=True).strip() != "Aqua":
        raise RuntimeError("XCTest requires the existing Aqua user session")
    xcode = subprocess.check_output(["xcodebuild", "-version"], text=True)
    if not xcode.startswith("Xcode 26.2\n"):
        raise RuntimeError("Regression checks require pinned Xcode 26.2")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {"group": group, "sourceRevision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "xcode": xcode.strip(),
        "status": "running", "steps": [], "signingAttempted": False, "testFlightUploaded": False,
        "trainedModelInferenceVerified": False, "physicalIPhoneVerified": False}
    path = OUTPUT / (group + "-checks.json")

    def save():
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        for index, step in enumerate(plan(group)):
            log_path = OUTPUT / (group + "-" + str(index).zfill(2) + ".log")
            env = os.environ.copy() | step["environment"]
            # All existing python3 commands use the same pinned control venv.
            env["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + env["PATH"]
            print("Running existing CI check: " + step["name"], flush=True)
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as stream:
                # Preserve pipefail: xcodebuild failure behind tee/tail is fatal.
                result = subprocess.run(["/bin/bash", "-e", "-o", "pipefail", "-c", step["body"]],
                    cwd=ROOT / step["directory"], env=env, stdout=stream, stderr=subprocess.STDOUT)
            record["steps"].append({key: step[key] for key in ("name", "workflow", "job", "bodySHA256", "directory")} |
                {"exitCode": result.returncode, "seconds": time.monotonic() - started, "log": log_path.name})
            save()
            if result.returncode:
                print(log_path.read_text(errors="replace")[-16000:], flush=True)
                raise RuntimeError("Existing CI check failed: " + step["name"])
            print("Passed existing CI check: " + step["name"], flush=True)
        record["status"] = "passed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(plan(args.group), indent=2))
    else:
        execute(args.group)
