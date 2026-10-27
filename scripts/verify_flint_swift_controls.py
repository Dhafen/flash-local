"""Compile source-identical Foundation protocol controls with official Swift.

These are portable Linux controls, NOT an Apple package test or iOS build.
Native actor/Metal/Network.framework are deliberately excluded, never stubbed.
Apple package/app gates still require xcodebuild before distribution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KIT = [ROOT / "Sources/MinirunKit/Serving" / name for name in [
    "LocalChatProtocol.swift", "LocalGenerationOptions.swift", "LocalToolProtocol.swift",
    "LocalFlintToolOutput.swift", "LocalAPIHTTP.swift", "LocalEngineLease.swift"]]
ENGINE = [ROOT / "Experiments/Flint/Sources/FlintEngine" / name for name in [
    "FlintPrefixPolicy.swift", "FlintTextStream.swift", "FlintWirePrompt.swift", "FlintNumerics.swift"]]
TESTS = [ROOT / "Tests/MinirunKitTests/LocalFlintProtocolTests.swift",
         ROOT / "Experiments/Flint/Tests/FlintEngineTests/FlintStreamTests.swift"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swiftc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([args.swiftc, "--version"], text=True).strip()
    flags = ["-swift-version", "5", "-strict-concurrency=complete", "-warnings-as-errors", "-g", "-I", str(out), "-L", str(out)]
    for module, files, links in [("MinirunKit", KIT, []), ("FlintEngine", ENGINE, ["-lMinirunKit"])]:
        subprocess.run([args.swiftc, *flags, "-enable-testing", "-emit-library", "-emit-module", "-module-name", module,
                        *map(str, files), *links, "-o", str(out / f"lib{module}.so")], check=True)
    entries, count = [], 0
    for path in TESTS:
        source = path.read_text(encoding="utf-8")
        cls = re.search(r"final class (\w+): XCTestCase", source)[1]
        methods = re.findall(r"func (test\w+)\(\)", source)
        count += len(methods)
        entries.append("testCase([" + ",".join(f'("{method}", {cls}.{method})' for method in methods) + "])")
    (out / "main.swift").write_text("import XCTest\nXCTMain([" + ",".join(entries) + "])\n", encoding="utf-8")
    executable = out / "flint-swift-controls"
    subprocess.run([args.swiftc, *flags, *map(str, TESTS), str(out / "main.swift"),
                    "-lMinirunKit", "-lFlintEngine", "-o", str(executable)], check=True)
    env = {**os.environ, "LD_LIBRARY_PATH": str(out) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    result = subprocess.run([str(executable)], text=True, capture_output=True, env=env)
    (out / "control.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout + result.stderr, end="")
    result.check_returncode()
    record = dict(schemaVersion=1, toolchain=version, platform="Linux Foundation controls",
                  testsPassed=count, applePackageTestsRun=False, iOSBuildVerified=False,
                  trainedInferenceExecuted=False, sourceIdentical=True,
                  sources={str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in KIT + ENGINE + TESTS})
    (out / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
