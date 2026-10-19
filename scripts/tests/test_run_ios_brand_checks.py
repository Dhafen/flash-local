"""Deterministic regressions for the fail-closed iOS visual delivery gate.

These tests need Python only; no installed Simulator and no signing secrets.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "run_ios_brand_checks.py"
spec = importlib.util.spec_from_file_location("run_ios_brand_checks", SCRIPT)
visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)


class IOSBrandGateTests(unittest.TestCase):
    def test_selects_first_available_phone_across_multiple_ios_runtimes(self):
        devices = {
            "devices": {
                "com.apple.CoreSimulator.SimRuntime.iOS-26-1": [
                    {"name": "iPhone 17 Pro Max", "udid": "FIRST", "isAvailable": True}
                ],
                "com.apple.CoreSimulator.SimRuntime.iOS-26-2": [
                    {"name": "iPhone 17 Pro Max", "udid": "SECOND", "isAvailable": True}
                ],
            }
        }
        with mock.patch.object(visual.subprocess, "run") as run:
            run.return_value.stdout = json.dumps(devices)
            self.assertEqual(visual.find_simulator(), "FIRST")

    def test_missing_simulator_fails_closed(self):
        with mock.patch.object(visual.subprocess, "run") as run:
            run.return_value.stdout = json.dumps({"devices": {"iOS": []}})
            with self.assertRaisesRegex(RuntimeError, "No available"):
                visual.find_simulator()

    def visual_gate_with_fake_xcodebuild(self, *, returncode, output, les_enabled=False):
        class CompletedPopen:
            pid = 999999
            def __init__(self, argv, stdout, stderr, start_new_session):
                self.argv = argv
                self.assertions = (
                    "-parallel-testing-enabled" in argv
                    and "NO" in argv
                    and sum(arg.startswith("-only-testing:") for arg in argv) == (4 if les_enabled else 3)
                )
                stdout.write(output)
                stdout.flush()
            def wait(self, timeout):
                return returncode

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(visual.os.environ, {"FLASH_LOCAL_ENABLE_LES": "1" if les_enabled else "0"}), mock.patch.object(visual.subprocess, "Popen", CompletedPopen):
                return visual.run_test_attempt(
                    "FAKE-SIM", seconds=120,
                    log=Path(d) / "ios-brand.log",
                    result=Path(d) / "ios-brand.xcresult",
                )

    def test_requires_every_requested_suite(self):
        three_of_four = (
            "** TEST SUCCEEDED **\n"
            "Test Suite 'ForkBrandRenderTests' passed\n"
            "Test Suite 'REAMChatRenderTests' passed\n"
            "Test Suite 'OrnithPerformanceRenderTests' passed\n"
        )
        self.assertFalse(
            self.visual_gate_with_fake_xcodebuild(returncode=0, output=three_of_four, les_enabled=True)
        )

    def test_standard_gate_accepts_all_three_without_les(self):
        three = "** TEST SUCCEEDED **\n" + "".join(
            f"Test Suite '{suite.rsplit('/', 1)[-1]}' passed\n"
            for suite in visual.TEST_SUITES
        )
        self.assertTrue(self.visual_gate_with_fake_xcodebuild(returncode=0, output=three))

    def test_failed_xcodebuild_cannot_pass_despite_suite_names(self):
        all_four = "** TEST SUCCEEDED **\n" + "".join(
            f"Test Suite '{suite.rsplit('/', 1)[-1]}' passed\n"
            for suite in visual.TEST_SUITES
        )
        self.assertFalse(
            self.visual_gate_with_fake_xcodebuild(returncode=65, output=all_four)
        )

    def test_all_four_pass_with_successful_xcodebuild(self):
        all_four = "** TEST SUCCEEDED **\n" + "".join(
            f"Test Suite '{suite.rsplit('/', 1)[-1]}' passed\n"
            for suite in (*visual.TEST_SUITES, visual.LES_SUITE)
        )
        self.assertTrue(
            self.visual_gate_with_fake_xcodebuild(returncode=0, output=all_four, les_enabled=True)
        )


if __name__ == "__main__":
    unittest.main()
