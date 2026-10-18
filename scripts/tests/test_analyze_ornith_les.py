import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_ornith_les import analyze, BACKENDS, CACHES, MODEL


def report():
    comparison = dict(positions=4, maximumAbsoluteError=.001, relativeL2Error=.0001,
        sameGreedy=True, routingDigest=19, routingDecisions=320)
    check = {key: copy.deepcopy(comparison) for key in ("noLookahead", "lookahead", "batchedPrefill")}
    check["modelSHA256"] = MODEL
    result = dict(schemaVersion=1, modelSHA256=MODEL, appBuild="112", promptID="ornith-lru-v1",
        platform="iOSDevice", deviceIdentifier="iPhone18,2", networkInferenceUsed=False,
        lowPowerModeEnabled=False, userDeclaredAirplaneMode=True, numericalCheck=check,
        contextCapacityTokens=4096, maximumObservedFootprintBytes=4_000_000_000, memoryAdmissions=[], measurements=[])
    for cache in CACHES:
        for backend, rate in zip(BACKENDS, (2, 3, 4, 1.5)):
            plan = None if backend == "existingGGUF" else dict(contextTokens=4096, processCeilingBytes=5_500_000_000,
                admittedProcessBytes=5_000_000_000, cacheSlotsPerLayer=0 if backend == "omniPageCache" else 20,
                lookaheadDepth=4 if backend == "omniLookahead" else 0, requestedLookaheadDepth=4 if backend == "omniLookahead" else 0)
            before = dict(metalCommandBuffers=100, metalTimestampedCommandBuffers=100,
                metalNanoseconds=1_000_000_000, metalDispatches=800, demandCacheHits=20, demandCacheMisses=30)
            after = dict(metalCommandBuffers=228, metalTimestampedCommandBuffers=228,
                metalNanoseconds=6_500_000_000, metalDispatches=1580, demandCacheHits=4020, demandCacheMisses=1030)
            run = dict(backend=backend, applicationCache=cache, osCacheControlled=False,
                decodeForwardPasses=128, generatedTokens=128, decodeForwardSeconds=128 / rate,
                promptTokens=50, prefillSeconds=10, decodeIOBytes=128_000,
                maximumObservedFootprintBytes=3_900_000_000, thermalStateAtStart=0, maximumThermalState=0,
                outputTokenSHA256="a" * 64, memoryPlan=plan,
                metricsAfterPrefill=before if plan else None, metricsAfter=after if plan else None)
            result["measurements"].append(dict(measurement=run, decodeForwardTokensPerSecond=rate,
                hasRequired128Tokens=True, lookaheadActive=plan["lookaheadDepth"] > 0 if plan else None))
    return result


class LESAnalysisTests(unittest.TestCase):
    def test_comparable_prompt_ranking_never_promotes_full_quality_or_app_selection(self):
        data = report()
        data["fullModelNumericsVerified"] = True
        result = analyze(data)
        self.assertTrue(result["shortNumericalGatePassed"])
        for value in result["comparisons"]:
            self.assertEqual(value["fastestComparableBackend"], "omniLookahead")
            self.assertEqual(value["speedupOverExisting"], 2)
        self.assertFalse(result["fullModelNumericsVerified"])
        self.assertFalse(result["reportAuthenticityVerified"])
        self.assertFalse(result["offlineExecutionVerified"])
        self.assertIsNone(result["recommendedAppBackend"])
        native = result["runs"][1]
        self.assertEqual(native["metalDecodeSeconds"], 5.5)
        self.assertEqual(native["demandExpertCacheHitRate"], .8)

    def test_legacy_failed_export_has_no_invented_baseline_or_speed(self):
        data = report()
        data.pop("numericalCheck"); data.pop("memoryAdmissions")
        data["measurements"] = []
        data["maximumObservedFootprintBytes"] = 2_279_560_368
        data["failure"] = "GGUF metadata budget exceeded: LES expert window cannot hold all selected experts plus prefetch"
        result = analyze(data)
        self.assertEqual(result["sampledPeakFootprintBytes"], 2_279_560_368)
        self.assertEqual(result["memoryAdmissions"], [])
        self.assertFalse(result["shortNumericalGatePassed"])
        self.assertTrue(all(value["fastestComparableBackend"] is None for value in result["comparisons"]))

    def test_legacy_short_gate_does_not_invent_batched_routing_evidence(self):
        result = analyze(report())
        self.assertTrue(result["shortNumericalGatePassed"])
        self.assertFalse(result["canonicalBatchedRoutingVerified"])

    def test_layer_and_matched_input_diagnostics_are_scalar_and_do_not_relax_the_gate(self):
        data = report()
        point = data["numericalCheck"]["noLookahead"]
        point.update(maximumAbsoluteError=3.12, relativeL2Error=.066,
            layerCheckpoints=dict(comparedPoints=500, missingPoints=0, shapeMismatches=0,
                maximumAbsoluteError=1, relativeL2Error=.01,
                firstDivergence=dict(position=0, stage=3, layer=0, maximumAbsoluteError=.1,
                    relativeL2Error=.002, values="PRIVATE_MARKER"), values="PRIVATE_MARKER"),
            projectionProbe=dict(samples=16, columns=2048, weightType=12, cpuInputType=15,
                backendMaximumAbsoluteError=.00001, cpuMaximumAbsoluteError=.01,
                backendRelativeL2Error=.000001, cpuRelativeL2Error=.001, input="PRIVATE_MARKER"))
        result = analyze(data)
        value = result["numericalComparisons"][0]
        self.assertEqual(value["layerCheckpoints"]["firstDivergence"]["stage"], 3)
        self.assertEqual(value["projectionProbe"]["samples"], 16)
        self.assertNotIn("PRIVATE_MARKER", json.dumps(result))
        self.assertFalse(result["shortNumericalGatePassed"])
        self.assertTrue(all(item["fastestComparableBackend"] is None for item in result["comparisons"]))
        point["layerCheckpoints"]["firstDivergence"]["layer"] = 40
        with self.assertRaises(ValueError): analyze(data)

    def test_canonical_trace_requires_all_three_matching_routes_and_counts(self):
        data = report()
        for key in ("noLookahead", "lookahead", "batchedPrefill"):
            data["numericalCheck"][key]["canonicalRoutingDigest"] = 2**64 - 1
        result = analyze(data)
        self.assertTrue(result["shortNumericalGatePassed"])
        self.assertTrue(result["canonicalBatchedRoutingVerified"])
        # The old order-dependent digest is intentionally different in batch.
        data["numericalCheck"]["batchedPrefill"]["routingDigest"] = 21
        self.assertTrue(analyze(data)["shortNumericalGatePassed"])
        for field, value in (("canonicalRoutingDigest", None), ("canonicalRoutingDigest", 5),
                             ("routingDecisions", None), ("routingDecisions", 319), ("routingDecisions", 0)):
            changed = copy.deepcopy(data)
            changed["numericalCheck"]["batchedPrefill"][field] = value
            result = analyze(changed)
            self.assertFalse(result["shortNumericalGatePassed"])
            self.assertFalse(result["canonicalBatchedRoutingVerified"])
            self.assertTrue(all(item["fastestComparableBackend"] is None for item in result["comparisons"]))
        data["numericalCheck"]["lookahead"]["canonicalRoutingDigest"] = 2**64
        with self.assertRaises(ValueError): analyze(data)

    def test_observed_success_does_not_replace_failed_standard_gate(self):
        data = report()
        observed = copy.deepcopy(data["numericalCheck"]["noLookahead"])
        data["numericalCheck"]["noLookahead"].update(maximumAbsoluteError=3.12, relativeL2Error=.066)
        data["numericalCheck"]["diagnostics"] = dict(instrumentedComparison=observed,
            instrumentationMaximumAbsoluteError=3.12, instrumentationRelativeL2Error=.066,
            referenceInstrumentationMaximumAbsoluteError=0, referenceInstrumentationRelativeL2Error=0,
            referenceCheckpointPayloadBytes=1024, values="PRIVATE_MARKER")
        result = analyze(data)
        self.assertEqual(result["numericalComparisons"][-1]["mode"], "diagnosticsOnly")
        self.assertEqual(result["numericalComparisons"][-1]["observerEffect"]["instrumentationMaximumAbsoluteError"], 3.12)
        self.assertFalse(result["shortNumericalGatePassed"])
        self.assertTrue(all(item["fastestComparableBackend"] is None for item in result["comparisons"]))
        self.assertNotIn("PRIVATE_MARKER", json.dumps(result))

    def test_rejected_plan_recovers_only_numeric_reservations_and_live_samples(self):
        data = report()
        data["memoryAdmissions"] = [dict(phase="numerics.prediction4.prefill1",
            beforeReclaim=dict(footprintBytes=550, availableBytes=5000),
            afterReclaim=dict(footprintBytes=350, availableBytes=5200), failure=
            "LES expert window cannot hold all 8 selected experts; window=7 requestedPrediction=4; "
            "baseline=350 available=5200 ceiling=5000 dense=1000 KV=20 SSM=30 reserve=1000 staging=100 bytesPerSlot=400")]
        sample = analyze(data)["memoryAdmissions"][0]
        self.assertEqual(sample["reclaimedFootprintBytes"], 200)
        self.assertEqual(sample["minimumProcessBytesForEightDemandSlots"], 5700)
        self.assertEqual(sample["marginAfterEightDemandSlots"], -700)
        self.assertEqual(sample["plan"]["requestedLookaheadDepth"], 4)
        data["memoryAdmissions"][0]["afterReclaim"]["footprintBytes"] = 650
        self.assertEqual(analyze(data)["memoryAdmissions"][0]["reclaimedFootprintBytes"], -100)

    def test_early_eos_missing_modes_and_duplicates_cannot_win(self):
        for kind in ("short", "missing"):
            data = report()
            if kind == "short":
                item = data["measurements"][2]
                item["measurement"].update(generatedTokens=9, decodeForwardPasses=9, decodeForwardSeconds=.1)
                item.update(decodeForwardTokensPerSecond=90, hasRequired128Tokens=False)
            else:
                data["measurements"].pop(2)
            result = analyze(data)
            self.assertIsNone(result["comparisons"][0]["fastestComparableBackend"])
            self.assertEqual(result["comparisons"][1]["fastestComparableBackend"], "omniLookahead")
        data = report(); data["measurements"].append(copy.deepcopy(data["measurements"][0]))
        with self.assertRaises(ValueError): analyze(data)

    def test_native_ownership_is_separate_from_footprint_and_private_data(self):
        data = report()
        data["memoryAdmissions"] = [dict(phase="numerics.prediction4.prefill1",
            beforeReclaim=dict(footprintBytes=2_200_000_000), afterReclaim=dict(footprintBytes=2_200_000_000),
            nativeMemory=dict(liveContexts=0, liveMetalBuffers=0, ownedMetalBytes=0,
                mappedMetalBytes=0, deviceAllocatedBytes=1_000_000, path="private", messages="private"))]
        sample = analyze(data)["memoryAdmissions"][0]
        self.assertEqual(sample["nativeMemory"]["liveMetalBuffers"], 0)
        self.assertEqual(sample["afterFootprintBytes"], 2_200_000_000)
        self.assertNotIn("private", json.dumps(sample))
        data["memoryAdmissions"][0]["nativeMemory"]["deviceAllocatedBytes"] = None
        self.assertIsNone(analyze(data)["memoryAdmissions"][0]["nativeMemory"]["deviceAllocatedBytes"])
        data["memoryAdmissions"][0]["nativeMemory"]["ownedMetalBytes"] = -1
        with self.assertRaises(ValueError): analyze(data)

    def test_all_reference_positions_routing_and_greedy_must_agree(self):
        for key, value in (("positions", 3), ("sameGreedy", False), ("maximumAbsoluteError", .02),
                           ("relativeL2Error", .002), ("routingDigest", 20), ("routingDecisions", 319)):
            data = report(); data["numericalCheck"]["lookahead"][key] = value
            result = analyze(data)
            self.assertFalse(result["shortNumericalGatePassed"])
            self.assertTrue(all(item["fastestComparableBackend"] is None for item in result["comparisons"]))
        data = report(); data["numericalCheck"]["lookahead"].pop("routingDigest")
        self.assertFalse(analyze(data)["shortNumericalGatePassed"])

    def test_hash_prompt_and_thermal_mismatch_block_comparison(self):
        for key, value in (("outputTokenSHA256", "b" * 64), ("promptTokens", 51),
                           ("maximumThermalState", 1), ("osCacheControlled", True)):
            data = report(); data["measurements"][2]["measurement"][key] = value
            self.assertIsNone(analyze(data)["comparisons"][0]["fastestComparableBackend"])
        data = report()
        for item in data["measurements"]:
            item["measurement"].update(thermalStateAtStart=2, maximumThermalState=2)
        self.assertTrue(all(item["fastestComparableBackend"] is None for item in analyze(data)["comparisons"]))

    def test_target_device_power_and_failure_are_not_ignored(self):
        for key, value in (("deviceIdentifier", "other"), ("platform", "iOSSimulator"),
                           ("lowPowerModeEnabled", True), ("networkInferenceUsed", True), ("failure", "stopped")):
            data = report(); data[key] = value
            self.assertTrue(all(item["fastestComparableBackend"] is None for item in analyze(data)["comparisons"]))

    def test_memory_excess_and_inactive_prediction_cannot_be_called_a_les_win(self):
        data = report(); item = data["measurements"][2]
        item["measurement"]["memoryPlan"]["lookaheadDepth"] = 0
        item["lookaheadActive"] = False
        result = analyze(data)
        self.assertIn("lookaheadWasInactive", result["runs"][2]["issues"])
        self.assertIsNone(result["comparisons"][0]["fastestComparableBackend"])
        for key, value in (("cacheSlotsPerLayer", 7), ("admittedProcessBytes", 5_600_000_000)):
            data = report(); data["measurements"][2]["measurement"]["memoryPlan"][key] = value
            self.assertIsNone(analyze(data)["comparisons"][0]["fastestComparableBackend"])
        data = report(); data["measurements"][2]["measurement"]["maximumObservedFootprintBytes"] = 5_600_000_000
        self.assertIsNone(analyze(data)["comparisons"][0]["fastestComparableBackend"])

    def test_partial_timestamps_are_unknown_gpu_time_and_counter_regression_is_rejected(self):
        data = report(); data["measurements"][2]["measurement"]["metricsAfter"]["metalTimestampedCommandBuffers"] -= 1
        value = analyze(data)["runs"][2]
        self.assertIsNone(value["metalDecodeSeconds"])
        self.assertFalse(value["fullMetalTimestampCoverage"])
        data["measurements"][2]["measurement"]["metricsAfter"]["metalDispatches"] = 799
        with self.assertRaises(ValueError): analyze(data)

    def test_context_capacity_and_cache_labels_must_match_actual_plan(self):
        for index, key, value in ((1, "contextTokens", 8192), (1, "lookaheadDepth", 4),
                                 (3, "cacheSlotsPerLayer", 8), (2, "requestedLookaheadDepth", 2),
                                 (2, "processCeilingBytes", 3_500_000_000)):
            data = report()
            item = data["measurements"][index]
            item["measurement"]["memoryPlan"][key] = value
            item["lookaheadActive"] = item["measurement"]["memoryPlan"]["lookaheadDepth"] > 0
            self.assertIsNone(analyze(data)["comparisons"][0]["fastestComparableBackend"])

    def test_invalid_or_contradictory_counts_do_not_generate_a_ranking(self):
        for key, value in (("decodeForwardPasses", True), ("decodeForwardSeconds", 0),
                           ("decodeForwardSeconds", float("nan")), ("decodeIOBytes", -1)):
            data = report(); data["measurements"][2]["measurement"][key] = value
            with self.assertRaises(ValueError): analyze(data)
        for key, value in (("decodeForwardTokensPerSecond", 99), ("hasRequired128Tokens", False), ("lookaheadActive", False)):
            data = report(); data["measurements"][2][key] = value
            with self.assertRaises(ValueError): analyze(data)

    def test_no_chat_tokens_private_paths_or_raw_failure_text_are_echoed(self):
        data = report(); data.update(appBuild="PRIVATE_MARKER", privatePath="PRIVATE_MARKER", failure="PRIVATE_MARKER")
        data["promptTokenIDs"] = [987654321]
        data["memoryAdmissions"] = [dict(phase="PRIVATE_MARKER", failure="PRIVATE_MARKER", plan=dict(privatePath="PRIVATE_MARKER"))]
        data["measurements"][0]["measurement"]["chat"] = "PRIVATE_MARKER"
        rendered = json.dumps(analyze(data), allow_nan=False)
        for value in ("PRIVATE_MARKER", "987654321", "promptTokenIDs", "privatePath"):
            self.assertNotIn(value, rendered)

    def test_cli_errors_do_not_echo_private_input_or_filename(self):
        script = Path(__file__).resolve().parents[1] / "analyze_ornith_les.py"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "PRIVATE_MARKER.json"
            path.write_text('{"PRIVATE_MARKER":', encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("PRIVATE_MARKER", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
