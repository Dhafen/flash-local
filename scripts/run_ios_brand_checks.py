#!/usr/bin/env python3
"""Bounded, fail-closed iOS Simulator visual tests for the existing Ornith CI.

A stalled simctl/xcodebuild must not silently consume the entire 120-minute
release job. Never skip visual tests or change their assertions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

TEST_SUITES = (
    "MinirunIOSTests/ForkBrandRenderTests",
    "MinirunIOSTests/REAMChatRenderTests",
    "MinirunIOSTests/OrnithPerformanceRenderTests",
    "MinirunIOSTests/OrnithLESRenderTests",
)
REQUIRED_SUITE = re.compile(r"ForkBrandRenderTests.*passed")


def find_simulator(name: str = "iPhone 17 Pro Max") -> str:
    proc = subprocess.run(
        ["xcrun", "simctl", "list", "devices", "available", "--json"],
        check=True, capture_output=True, text=True, timeout=90,
    )
    devices = json.loads(proc.stdout)["devices"]
    matches = [
        dev["udid"]
        for runtime in devices.values()
        for dev in runtime
        if dev.get("name") == name and dev.get("isAvailable")
    ]
    if not matches:
        raise RuntimeError(f"No available {name} in simctl device list")
    # A runner can expose the same device name under several iOS runtimes.
    # Preserve the previous CI's first-available choice rather than rejecting it.
    return matches[0]


def bound_run(argv: list[str], *, seconds: int, check: bool = True) -> int:
    result = subprocess.run(argv, check=False, timeout=seconds)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {argv!r}")
    return result.returncode


def prepare_simulator(udid: str) -> None:
    # Boot can legitimately return nonzero if it is already running.
    bound_run(["xcrun", "simctl", "boot", udid], seconds=90, check=False)
    bound_run(["xcrun", "simctl", "bootstatus", udid, "-b"], seconds=240)


def run_test_attempt(udid: str, *, seconds: int, log: Path, result: Path) -> bool:
    if result.exists():
        shutil.rmtree(result)
    cmd = [
        "xcodebuild", "test",
        "-project", "Apps/Minirun/Minirun.xcodeproj",
        "-scheme", "Minirun-iOS",
        "-destination", f"platform=iOS Simulator,id={udid}",
        "-destination-timeout", "90",
        "-parallel-testing-enabled", "NO",
        "-maximum-concurrent-test-simulator-destinations", "1",
        "-resultBundlePath", str(result),
        "-skipPackagePluginValidation", "-skipMacroValidation",
        "CODE_SIGNING_ALLOWED=NO",
    ]
    cmd += [f"-only-testing:{suite}" for suite in TEST_SUITES]
    print(f"Running all {len(TEST_SUITES)} visual suites (deadline: {seconds}s)", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(
            cmd, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    code = proc.wait(timeout=45)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f"Simulator visual tests running {int(elapsed)}s / {seconds}s", flush=True)
                    if elapsed >= seconds:
                        raise TimeoutError(f"xcodebuild exceeded {seconds}s")
        except (TimeoutError, KeyboardInterrupt):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            print("Timed-out simulator test process group terminated", flush=True)
            return False
    content = log.read_text(encoding="utf-8", errors="replace")
    # Pass iff xcodebuild passed AND the required brand suite actually ran.
    okay = code == 0 and bool(REQUIRED_SUITE.search(content)) and "** TEST SUCCEEDED **" in content
    print(f"Visual test completed: rc={code}, brand-suite={bool(REQUIRED_SUITE.search(content))}, pass={okay}", flush=True)
    if not okay:
        print("Last 35 log lines:", flush=True)
        print("\n".join(content.splitlines()[-35:]), flush=True)
    return okay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-seconds", type=int, default=1500)
    ap.add_argument("--attempts", type=int, default=2)
    args = ap.parse_args()
    if not 60 <= args.timeout_seconds <= 2400 or args.attempts not in (1, 2):
        ap.error("Use 60..2400 seconds and 1 or 2 attempts")
    success = False
    for attempt in range(1, args.attempts + 1):
        try:
            udid = find_simulator()
            print(f"iPhone simulator visual gate attempt {attempt}/{args.attempts}: {udid}", flush=True)
            prepare_simulator(udid)
            log = Path("ios-brand.log")
            result = Path("ios-brand.xcresult")
            success = run_test_attempt(udid, seconds=args.timeout_seconds, log=log, result=result)
            if success:
                break
        except (subprocess.TimeoutExpired, OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"Simulator attempt {attempt} failed: {exc}", flush=True)
        if attempt < args.attempts:
            if Path("ios-brand.log").exists():
                shutil.copyfile("ios-brand.log", f"ios-brand-attempt-{attempt}.log")
            try:
                bound_run(["xcrun", "simctl", "shutdown", udid], seconds=60, check=False)
            except (subprocess.TimeoutExpired, RuntimeError, OSError, UnboundLocalError):
                pass
    if not success:
        print("FAILED: iOS visual gate never passed; no archive or delivery is allowed.", flush=True)
        return 1
    bound_run(
        ["xcrun", "xcresulttool", "export", "attachments", "--path",
         "ios-brand.xcresult", "--output-path", "brand-ios"],
        seconds=180,
    )
    print("PASSED: iOS visual tests, brand evidence, and artifact export", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
