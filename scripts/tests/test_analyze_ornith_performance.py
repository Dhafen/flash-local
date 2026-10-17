import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_ornith_performance import analyze


def gguf():
    return {
        "appBuild": "7", "modelSourceURL": "https://example.test/model.gguf",
        "contextTokens": 4096, "platform": "iOSDevice",
        "generation": {"decodeForwardPasses": 2, "decodeForwardSeconds": 1.0,
            "decodeIOBytesRead": 200, "prefillSeconds": 4.0,
            "evaluatedPromptTokens": 8, "reusedPromptTokens": 12,
            "maximumObservedFootprintBytes": 1000,
            "decodeProjections": {"computeSeconds": 0.5}},
    }


def qpack():
    report = gguf()
    report.pop("modelSourceURL")
    report["inferenceBackend"] = "SwiftletCore Metal / local QPack"
    report["generation"] = {
        "contextTokens": 8192, "prefillSeconds": 10.0,
        "forwardMetricsIncludeInterruptedWork": False,
        "forwards": [
            {"phase": "prefill", "completed": True, "wallSeconds": 10.0},
            {"phase": "decode", "inputTokens": 1, "completed": True,
             "wallSeconds": 0.5, "expertReturnedReadBytes": 100,
             "dispatches": 32, "commandBuffers": 3,
             "expertCacheHits": 6, "expertCacheMisses": 2,
             "metalGPUSeconds": 0.2, "blockingWaitSeconds": 0.3},
        ],
    }
    return report


class PerformanceAnalysisTests(unittest.TestCase):
    def test_gguf_build_does_not_imply_qpack_and_targets_are_arithmetic(self):
        report = gguf()
        report["appBuild"] = "94"
        result = analyze(report)
        self.assertEqual(result["backend"], "GGUF-packed")
        self.assertEqual(result["decode"]["meanMillisecondsPerForward"], 500)
        self.assertEqual(result["targets"][0]["requiredSpeedup"], 5)
        self.assertEqual(result["targets"][0]["requiredReturnedReadBytesPerSecondAtUnchangedReadVolume"], 1000)
        self.assertEqual(result["prefill"]["evaluatedTokensPerSecond"], 2)
        self.assertFalse(result["evidence"]["sustainedTargetEstablished"])

    def test_qpack_uses_decode_only_and_complete_forward_count(self):
        report = qpack()
        report["generation"]["forwards"].append({"phase": "decode", "completed": False, "wallSeconds": 8.0})
        result = analyze(report)
        self.assertEqual(result["decode"]["wallSeconds"], 0.5)
        self.assertEqual(result["decode"]["completedForwards"], 1)
        self.assertEqual(result["decode"]["breakdown"]["expertCacheHitRate"], 0.75)
        self.assertEqual(result["contextCapacityTokens"], 8192)

    def test_partial_gpu_timing_is_unknown_not_zero_or_partial_sum(self):
        report = qpack()
        other = copy.deepcopy(report["generation"]["forwards"][1])
        other["metalGPUSeconds"] = None
        report["generation"]["forwards"].append(other)
        result = analyze(report)
        self.assertIsNone(result["decode"]["breakdown"]["metalGPUSeconds"])
        self.assertEqual(result["decode"]["breakdown"]["blockingWaitSeconds"], 0.6)

    def test_no_chat_tokens_paths_or_arbitrary_fields_in_output(self):
        report = gguf()
        report["privatePath"] = "PRIVATE_MARKER"
        report["appBuild"] = "PRIVATE_MARKER"
        report["generation"]["promptTokenIDs"] = [987654321]
        report["generation"]["generatedTokenIDs"] = [987654321]
        report["generation"]["chat"] = "PRIVATE_MARKER"
        rendered = json.dumps(analyze(report))
        for forbidden in ("PRIVATE_MARKER", "987654321", "promptTokenIDs", "generatedTokenIDs"):
            self.assertNotIn(forbidden, rendered)

    def test_refuses_wrong_backend_marker_or_multitoken_decode(self):
        report = qpack()
        report.pop("inferenceBackend")
        with self.assertRaises(ValueError):
            analyze(report)
        report = qpack()
        report["generation"]["forwards"][1]["inputTokens"] = 2
        with self.assertRaises(ValueError):
            analyze(report)

    def test_invalid_or_inconsistent_measurements_do_not_make_speed_claims(self):
        for field, value in (("decodeForwardPasses", 0), ("decodeForwardPasses", True),
                             ("decodeForwardPasses", 1.5), ("decodeForwardSeconds", float("nan")),
                             ("decodeForwardSeconds", 0), ("decodeIOBytesRead", -1),
                             ("decodeForwardTokensPerSecond", 99)):
            with self.subTest(field=field, value=value):
                report = gguf()
                report["generation"][field] = value
                with self.assertRaises(ValueError):
                    analyze(report)

    def test_unsupported_and_no_decode_reports_fail(self):
        with self.assertRaises(ValueError):
            analyze({"generation": {}})
        report = qpack()
        report["generation"]["forwards"] = []
        with self.assertRaises(ValueError):
            analyze(report)

    def test_missing_metrics_are_not_fabricated(self):
        report = gguf()
        report["generation"].pop("decodeProjections")
        result = analyze(report)
        self.assertIsNone(result["decode"]["breakdown"]["computeSeconds"])
        self.assertIsNone(result["decode"]["breakdown"]["evaluationBatches"])
        self.assertFalse(result["evidence"]["qualityEquivalenceEstablished"])


if __name__ == "__main__":
    unittest.main()
