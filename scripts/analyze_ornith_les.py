#!/usr/bin/env python3
"""Analyze a local LES export; print whitelisted aggregates, never chat or paths.

Rankings describe this single prompt/report only. They never select an app
backend or establish full model quality, offline authenticity or sustained speed.
"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_ornith_performance import number


MODEL = "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"
BACKENDS = ("existingGGUF", "omniCache", "omniLookahead", "omniPageCache")
CACHES = ("initiallyEmpty", "warm")
CEILING = 5_500_000_000
PLAN_FIELDS = ("contextTokens", "processCeilingBytes", "baselineFootprintBytes", "baselineAvailableBytes",
    "denseBytes", "kvBytes", "ssmBytes", "runtimeReserveBytes", "prefillAllocationBytes",
    "expertBytesPerSlot", "expertAllocationBytes", "admittedProcessBytes",
    "cacheSlotsPerLayer", "lookaheadDepth", "requestedLookaheadDepth")
FAILURE_FIELDS = dict(baseline="baselineFootprintBytes", available="baselineAvailableBytes",
    ceiling="processCeilingBytes", dense="denseBytes", KV="kvBytes", SSM="ssmBytes",
    reserve="runtimeReserveBytes", staging="prefillAllocationBytes", bytesPerSlot="expertBytesPerSlot",
    window="totalWindowSlots", requestedPrediction="requestedLookaheadDepth")
NATIVE_MEMORY_FIELDS = ("liveContexts", "liveMetalBuffers", "ownedMetalBytes", "mappedMetalBytes", "deviceAllocatedBytes")


def optional_number(obj, key, *, integer=True):
    value = obj.get(key)
    return None if value is None else number(value, key, integer=integer)


def plan_numbers(plan):
    if not isinstance(plan, dict):
        return {}
    return {key: optional_number(plan, key) for key in PLAN_FIELDS if key in plan}


def memory_admission(item):
    if not isinstance(item, dict):
        raise ValueError("invalid memory admission")
    phase = item.get("phase")
    known_phase = isinstance(phase, str) and re.fullmatch(
        r"preparation|numerics\.prediction\d{1,2}\.prefill\d{1,2}|benchmark\.(?:existingGGUF|omniCache|omniLookahead|omniPageCache)", phase)
    before, after = item.get("beforeReclaim") or {}, item.get("afterReclaim") or {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("invalid footprint sample")
    before_bytes = optional_number(before, "footprintBytes")
    after_bytes = optional_number(after, "footprintBytes")
    plan = plan_numbers(item.get("plan"))
    failure = item.get("failure")
    # Failed admission has no plan. Recover only the published numeric fields;
    # arbitrary error text, local paths and unknown phase names stay private.
    if isinstance(failure, str):
        for key, value in re.findall(r"(?:^|\s)([A-Za-z]+)=(\d{1,20})(?=\s|;|$)", failure):
            if key in FAILURE_FIELDS and int(value) <= 2**64 - 1:
                plan.setdefault(FAILURE_FIELDS[key], int(value))
    required = ("baselineFootprintBytes", "denseBytes", "kvBytes", "ssmBytes",
        "runtimeReserveBytes", "prefillAllocationBytes", "expertBytesPerSlot", "processCeilingBytes")
    minimum = sum(plan[k] for k in required[:-2]) + 8 * plan["expertBytesPerSlot"] \
        if all(plan.get(k) is not None for k in required) else None
    native = item.get("nativeMemory")
    if native is not None and not isinstance(native, dict):
        raise ValueError("invalid native memory sample")
    native = {key: optional_number(native, key) for key in NATIVE_MEMORY_FIELDS if key in native} \
        if native is not None else None
    return dict(phase=phase if known_phase else None, failureReported=failure is not None,
        beforeFootprintBytes=before_bytes, afterFootprintBytes=after_bytes,
        reclaimedFootprintBytes=before_bytes - after_bytes if before_bytes is not None and after_bytes is not None else None,
        availableAfterBytes=optional_number(after, "availableBytes"), plan=plan,
        nativeMemory=native,
        minimumProcessBytesForEightDemandSlots=minimum,
        marginAfterEightDemandSlots=plan["processCeilingBytes"] - minimum if minimum is not None else None)


def numerical_gate(check):
    if not isinstance(check, dict) or check.get("modelSHA256") != MODEL:
        return False
    for key in ("noLookahead", "lookahead", "batchedPrefill"):
        value = check.get(key)
        if not isinstance(value, dict) or value.get("sameGreedy") is not True:
            return False
        if number(value.get("positions"), "positions", integer=True) != 4:
            return False
        if number(value.get("maximumAbsoluteError"), "maximumAbsoluteError") > .01 or \
                number(value.get("relativeL2Error"), "relativeL2Error") > .001:
            return False
    off, on = check["noLookahead"], check["lookahead"]
    if any(value.get(key) is None for value in (off, on) for key in ("routingDigest", "routingDecisions")):
        return False
    return (number(off.get("routingDecisions"), "routingDecisions", integer=True) > 0 and
        off["routingDecisions"] == on.get("routingDecisions") and
        number(off.get("routingDigest"), "routingDigest", integer=True) ==
        number(on.get("routingDigest"), "routingDigest", integer=True))


def delta(before, after, key):
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    first, last = optional_number(before, key), optional_number(after, key)
    if first is None or last is None:
        return None
    if last < first:
        raise ValueError("counter regressed")
    return last - first


def measurement(item):
    if not isinstance(item, dict) or not isinstance(item.get("measurement"), dict):
        raise ValueError("invalid measurement")
    run = item["measurement"]
    backend, cache = run.get("backend"), run.get("applicationCache")
    if backend not in BACKENDS or cache not in CACHES:
        raise ValueError("unrecognized benchmark variant")
    count = number(run.get("decodeForwardPasses"), "decodeForwardPasses", integer=True)
    generated = number(run.get("generatedTokens"), "generatedTokens", integer=True)
    declared_length = item.get("hasRequired128Tokens")
    if declared_length is not None and (not isinstance(declared_length, bool) or
            declared_length != (generated >= 128 and count >= 128)):
        raise ValueError("length status mismatch")
    seconds = number(run.get("decodeForwardSeconds"), "decodeForwardSeconds", positive=count > 0)
    rate = count / seconds if count else 0
    declared = item.get("decodeForwardTokensPerSecond")
    if declared is not None and not math.isclose(number(declared, "declared rate"), rate, rel_tol=1e-6):
        raise ValueError("decode rate mismatch")
    digest = run.get("outputTokenSHA256")
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("invalid output digest")
    prompt = number(run.get("promptTokens"), "promptTokens", integer=True, positive=True)
    prefill = number(run.get("prefillSeconds"), "prefillSeconds", positive=True)
    reads = number(run.get("decodeIOBytes"), "decodeIOBytes", integer=True)
    peak = number(run.get("maximumObservedFootprintBytes"), "maximumObservedFootprintBytes", integer=True, positive=True)
    start = number(run.get("thermalStateAtStart"), "thermalStateAtStart", integer=True)
    maximum = number(run.get("maximumThermalState"), "maximumThermalState", integer=True)
    if not 0 <= start <= maximum <= 3:
        raise ValueError("invalid thermal states")
    plan = plan_numbers(run.get("memoryPlan"))
    active_depth = plan.get("lookaheadDepth")
    if item.get("lookaheadActive") is not None:
        if not isinstance(item["lookaheadActive"], bool) or item["lookaheadActive"] != (active_depth is not None and active_depth > 0):
            raise ValueError("lookahead status mismatch")
    issues = []
    if generated < 128 or count < 128 or count != generated:
        issues.append("incomplete128TokenRun")
    if run.get("osCacheControlled") is not False:
        issues.append("unsupportedOSCacheControlClaim")
    if peak > CEILING:
        issues.append("observedMemoryExceedsBudget")
    if backend != "existingGGUF":
        ceiling, admitted = plan.get("processCeilingBytes"), plan.get("admittedProcessBytes")
        if ceiling is None or admitted is None or not 0 < admitted <= ceiling <= CEILING:
            issues.append("memoryAdmissionNotEstablished")
        elif peak > ceiling:
            issues.append("observedMemoryExceedsNativeCeiling")
        slots = plan.get("cacheSlotsPerLayer")
        if backend != "omniPageCache" and (slots is None or slots < 8):
            issues.append("eightDemandSlotsNotEstablished")
        requested = plan.get("requestedLookaheadDepth")
        if (slots is None or slots > 256 or active_depth is None or active_depth > 16 or
                requested is not None and not active_depth <= requested <= 16 or
                backend != "omniLookahead" and active_depth != 0 or
                backend == "omniPageCache" and slots != 0):
            issues.append("benchmarkModeAndGeometryDisagree")
        if backend == "omniLookahead" and not active_depth:
            issues.append("lookaheadWasInactive")
    before, after = run.get("metricsAfterPrefill"), run.get("metricsAfter")
    buffers = delta(before, after, "metalCommandBuffers")
    timed = delta(before, after, "metalTimestampedCommandBuffers")
    gpu_ns = delta(before, after, "metalNanoseconds")
    if buffers is not None and timed is not None and timed > buffers:
        raise ValueError("timestamp count exceeds command buffers")
    covered = buffers is not None and buffers > 0 and timed == buffers and gpu_ns is not None
    hits, misses = delta(before, after, "demandCacheHits"), delta(before, after, "demandCacheMisses")
    return dict(backend=backend, applicationCache=cache, generatedTokens=generated,
        completedForwards=count, forwardsPerSecond=rate, prefillTokensPerSecond=prompt / prefill,
        promptTokens=prompt, decodeReturnedReadBytes=reads, decodeReadBytesPerForward=reads / count if count else None,
        maximumObservedFootprintBytes=peak, thermalStateAtStart=start, maximumThermalState=maximum,
        contextCapacityTokens=plan.get("contextTokens"),
        activeLookaheadDepth=active_depth, requestedLookaheadDepth=plan.get("requestedLookaheadDepth"),
        metalDecodeDispatches=delta(before, after, "metalDispatches"),
        metalCommandBuffers=buffers, timestampedCommandBuffers=timed,
        metalDecodeSeconds=gpu_ns / 1e9 if covered else None, fullMetalTimestampCoverage=covered,
        demandExpertCacheHitRate=hits / (hits + misses) if hits is not None and misses is not None and hits + misses else None,
        outputTokenSHA256=digest, issues=issues)


def analyze(report):
    if not isinstance(report, dict) or type(report.get("schemaVersion")) is not int or report["schemaVersion"] != 1 or report.get("modelSHA256") != MODEL:
        raise ValueError("unsupported LES report")
    values, admissions = report.get("measurements"), report.get("memoryAdmissions", [])
    if not isinstance(values, list) or len(values) > 8 or not isinstance(admissions, list) or len(admissions) > 64:
        raise ValueError("invalid report cardinality")
    runs = [measurement(item) for item in values]
    keys = [(run["backend"], run["applicationCache"]) for run in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate benchmark variant")
    gate = numerical_gate(report.get("numericalCheck"))
    context = number(report.get("contextCapacityTokens"), "contextCapacityTokens", integer=True)
    if not 32 <= context <= 65536:
        raise ValueError("unsupported context capacity")
    for run in runs:
        if run["backend"] != "existingGGUF" and run["contextCapacityTokens"] != context:
            run["issues"].append("contextCapacityMismatch")
    global_issues = []
    for condition, issue in [
        (report.get("failure") is None, "reportContainsFailure"),
        (gate, "shortNumericalGateNotPassed"),
        (report.get("platform") == "iOSDevice" and report.get("deviceIdentifier") == "iPhone18,2", "targetPhoneNotEstablished"),
        (report.get("promptID") == "ornith-lru-v1", "benchmarkPromptNotEstablished"),
        (report.get("networkInferenceUsed") is False, "localRunnerNotEstablished"),
        (report.get("lowPowerModeEnabled") is False, "lowPowerModeNotConfirmedOff"),
    ]:
        if not condition:
            global_issues.append(issue)
    comparisons = []
    for cache in CACHES:
        selected = [run for run in runs if run["applicationCache"] == cache]
        issues = list(global_issues)
        if {run["backend"] for run in selected} != set(BACKENDS):
            issues.append("missingBenchmarkModes")
        if any(run["issues"] for run in selected):
            issues.append("invalidOrIncompleteRun")
        if len({run["outputTokenSHA256"] for run in selected}) != 1:
            issues.append("outputTokenSequencesDifferOrMissing")
        if len({run["promptTokens"] for run in selected}) != 1:
            issues.append("promptLengthsDifferOrMissing")
        thermals = {(run["thermalStateAtStart"], run["maximumThermalState"]) for run in selected}
        if len(thermals) != 1 or any(maximum >= 2 for _, maximum in thermals):
            issues.append("thermalConditionsNotComparable")
        fastest = max(selected, key=lambda run: run["forwardsPerSecond"]) if selected and not issues else None
        baseline = next((run for run in selected if run["backend"] == "existingGGUF"), None)
        comparisons.append(dict(applicationCache=cache, issues=issues,
            fastestComparableBackend=fastest["backend"] if fastest else None,
            speedupOverExisting=fastest["forwardsPerSecond"] / baseline["forwardsPerSecond"] if fastest else None))
    build = report.get("appBuild")
    build = build if isinstance(build, str) and re.fullmatch(r"[0-9]{1,16}", build) else None
    return dict(schemaVersion=1, appBuild=build, contextCapacityTokens=context, failedReport=report.get("failure") is not None,
        sampledPeakFootprintBytes=optional_number(report, "maximumObservedFootprintBytes"),
        shortNumericalGatePassed=gate, fullModelNumericsVerified=False,
        memoryAdmissions=[memory_admission(item) for item in admissions], runs=runs, comparisons=comparisons,
        recommendedAppBackend=None, reportAuthenticityVerified=False, offlineExecutionVerified=False,
        scope="single fixed prompt; four reference positions; no automatic backend selection",
        notes=["Observed peak is not the live baseline at an admission failure.",
            "Empty application cache does not establish a cold OS page cache.",
            "Returned source reads omit dense mmap faults and do not measure physical flash traffic.",
            "Metal durations can overlap CPU work; partial GPU timestamp sums are not full GPU time.",
            "A ranking within one report does not establish sustained speed or general quality equivalence."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        if args.report.stat().st_size > 16 << 20:
            raise ValueError("report exceeds local input limit")
        result = analyze(json.loads(args.report.read_text(encoding="utf-8-sig")))
    except (ValueError, OSError, TypeError, AttributeError, KeyError, OverflowError):
        parser.exit(2, "Cannot analyze this LES report.\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
