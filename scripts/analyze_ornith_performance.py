#!/usr/bin/env python3
"""Read an app export locally; emit numeric aggregates, never chat/token arrays.

This diagnoses a run, not benchmark validity, numerical parity, or sustained
phone performance. Forward throughput is not user-visible output throughput.
"""
import argparse
import json
import math
from pathlib import Path


def number(value, name, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing/invalid {name}")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    if integer and value != int(value):
        raise ValueError(f"non-integral {name}")
    return int(value) if integer else value


def optional_number(obj, key):
    value = obj.get(key)
    return None if value is None else number(value, key)


def analyze(report):
    generation = report.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("missing generation")
    notes = [
        "Decode forwards/s is not end-to-end generated output tokens/s.",
        "Returned read bytes do not measure physical flash-controller traffic.",
        "A single export does not establish a controlled A/B or sustained speed.",
    ]
    breakdown = {}
    kernel_mode = None
    if "forwards" in generation:
        if report.get("inferenceBackend") != "SwiftletCore Metal / local QPack":
            raise ValueError("QPack forward records require the QPack backend marker")
        backend = "SwiftletCore-QPack"
        forwards = generation["forwards"]
        if not isinstance(forwards, list) or any(not isinstance(f, dict) for f in forwards):
            raise ValueError("invalid forwards")
        decode = [f for f in forwards if f.get("phase") == "decode" and f.get("completed") is True]
        if any(f.get("inputTokens") != 1 for f in decode):
            raise ValueError("multi-token decode needs an accepted-output metric")
        count = len(decode)
        modes = {f.get("kernelMode") for f in decode}
        if modes == {None} or not modes:
            kernel_mode = None  # Legacy export; never infer mode from build.
        elif len(modes) == 1 and modes <= {"original", "groupedExperts"}:
            kernel_mode = modes.pop()
        else:
            raise ValueError("mixed or unsupported decode kernel modes")
        declared_mode = report.get("kernelMode")
        if declared_mode is not None and declared_mode != kernel_mode:
            raise ValueError("kernel mode disagrees with forward records")
        seconds = sum(number(f.get("wallSeconds"), "wallSeconds", positive=True) for f in decode)
        read_bytes = sum(number(f.get("expertReturnedReadBytes"), "expertReturnedReadBytes", integer=True) for f in decode)
        for key in ("dispatches", "commandBuffers", "expertCacheHits", "expertCacheMisses"):
            breakdown[key] = sum(number(f.get(key), key, integer=True) for f in decode)
        for key in ("encodeSeconds", "blockingWaitSeconds", "expertFetchWallSeconds", "metalGPUSeconds"):
            values = [optional_number(f, key) for f in decode]
            breakdown[key] = sum(values) if values and all(v is not None for v in values) else None
        hits, misses = breakdown["expertCacheHits"], breakdown["expertCacheMisses"]
        breakdown["expertCacheHitRate"] = hits / (hits + misses) if hits + misses else None
        notes.append("GPU duration and CPU blocking wait overlap; do not add them. Concurrent read durations are not elapsed I/O wait.")
        if generation.get("forwardMetricsIncludeInterruptedWork") is False:
            notes.append("Forward aggregates exclude interrupted work; elapsed generation time can include it.")
    elif "decodeForwardPasses" in generation:
        if "inferenceBackend" in report or "modelSourceURL" not in report or ".gguf" not in report["modelSourceURL"]:
            raise ValueError("ambiguous GGUF backend")
        backend = "GGUF-packed"
        count = number(generation.get("decodeForwardPasses"), "decodeForwardPasses", integer=True)
        seconds = number(generation.get("decodeForwardSeconds"), "decodeForwardSeconds")
        read_bytes = number(generation.get("decodeIOBytesRead"), "decodeIOBytesRead", integer=True)
        projections = generation.get("decodeProjections", {})
        for key in ("computeSeconds", "waitOnIOSeconds", "evaluationBatches"):
            breakdown[key] = optional_number(projections, key)
        notes.append("GGUF computeSeconds includes host evaluation/synchronization; it is not a GPU timestamp measurement.")
        notes.append("This report does not measure the optional Swiftlet/QPack engine, even when the app build contains it.")
    else:
        raise ValueError("unrecognized generation schema")
    number(count, "completed decode forwards", integer=True, positive=True)
    number(seconds, "decode seconds", positive=True)
    rate = count / seconds
    declared_rate = generation.get("decodeForwardTokensPerSecond", report.get("decodeForwardTokensPerSecond"))
    if declared_rate is not None and not math.isclose(number(declared_rate, "declared rate"), rate, rel_tol=1e-6):
        raise ValueError("reported decode rate disagrees with completed forwards and time")
    context = generation.get("contextTokens", report.get("contextTokens"))
    prompt_evaluated = optional_number(generation, "evaluatedPromptTokens")
    prefill_seconds = optional_number(generation, "prefillSeconds")
    if count < 128:
        notes.append("Fewer than 128 decode forwards: smoke test, not the planned throughput benchmark.")
    footprint = optional_number(generation, "maximumObservedFootprintBytes")
    build = report.get("appBuild")
    if not isinstance(build, str) or not build.isascii() or not build.isdigit():
        build = None
    return {
        "schemaVersion": 1,
        "backend": backend,
        "kernelMode": kernel_mode,
        "appBuild": build,
        "contextCapacityTokens": number(context, "context capacity", integer=True, positive=True),
        "decode": {
            "completedForwards": count,
            "wallSeconds": seconds,
            "forwardsPerSecond": rate,
            "meanMillisecondsPerForward": seconds * 1000 / count,
            "returnedReadBytes": read_bytes,
            "returnedReadBytesPerForward": read_bytes / count,
            "observedReturnedBytesPerDecodeWallSecond": read_bytes / seconds,
            "breakdown": breakdown,
        },
        "prefill": {
            "seconds": prefill_seconds,
            "evaluatedTokens": prompt_evaluated,
            "reusedTokens": optional_number(generation, "reusedPromptTokens"),
            "evaluatedTokensPerSecond": prompt_evaluated / prefill_seconds if prompt_evaluated is not None and prefill_seconds else None,
        },
        "maximumObservedFootprintBytes": footprint,
        "targets": [{
            "forwardsPerSecond": target,
            "budgetMillisecondsPerForward": 1000 / target,
            "requiredSpeedup": target / rate,
            "requiredReturnedReadBytesPerSecondAtUnchangedReadVolume": target * read_bytes / count,
        } for target in (10, 15)],
        "evidence": {
            "reportedPhysicalDevice": report.get("platform") == "iOSDevice",
            "reportedAirplaneModeVerified": report.get("airplaneModeVerified") is True,
            "reportedFullModelNumericsVerified": report.get("fullModelNumericsVerified") is True,
            "reportAuthenticityVerified": False,
            "sustainedTargetEstablished": False,
            "qualityEquivalenceEstablished": False,
        },
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        if args.report.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("report exceeds the 16 MiB local input limit")
        result = analyze(json.loads(args.report.read_text(encoding="utf-8-sig")))
    except (ValueError, OSError, TypeError, AttributeError) as exc:
        # Avoid echoing input contents or private file paths in diagnostics.
        parser.exit(2, f"Cannot analyze this report ({type(exc).__name__}).\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
