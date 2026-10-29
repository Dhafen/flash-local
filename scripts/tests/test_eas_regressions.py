import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("eas_regressions", ROOT / "scripts/run_eas_regressions.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class EASRegressionBoundaryTests(unittest.TestCase):
    def test_environment_resolves_only_workspace_and_rejects_credentials(self):
        root = Path("/repo")
        self.assertEqual(checks.resolve_environment({"TEST_RUNNER_ORNITH_MLX_TOKENIZER_FIXTURE":
            "${{ github.workspace }}/fixture"}, root),
            {"TEST_RUNNER_ORNITH_MLX_TOKENIZER_FIXTURE": str(root) + "/fixture"})
        for values in ({"EXPO_TOKEN": "fixture"}, {"APP_STORE_API_KEY": "fixture"},
                       {"VALUE": "${{ secrets.KEY }}"}, {"VALUE": True}):
            with self.assertRaises(ValueError):
                checks.resolve_environment(values, ROOT)

    def test_non_apple_host_never_claims_regression_execution(self):
        with patch.object(checks.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "actual EAS Apple host"):
                checks.execute("qpack")


@unittest.skipUnless(importlib.util.find_spec("yaml"), "EAS-only PyYAML dependency not installed on this host")
class EASWorkflowSourceTests(unittest.TestCase):
    def test_selected_bodies_keep_the_existing_critical_assertions(self):
        qpack = checks.plan("qpack")
        self.assertEqual(len(qpack), 6)
        body = "\n".join(step["body"] for step in qpack)
        self.assertIn("1e348e7b61289d70ba5bde2c212bcfa7867f4323", body)
        self.assertIn("check_qpack_validation_log.py", body)
        self.assertIn("SwiftletSessionTests TextGeneratorTests", body)
        self.assertIn("ORNITH_MLX_TOKENIZER_PARITY", body)
        self.assertNotIn("sudo xcode-select", body)
        regression = checks.plan("regression")
        self.assertTrue(any("ORNITH_DFLASH_GRAPH" in step["body"] for step in regression))
        self.assertTrue(any("ORNITH_PAIRED_GRAPH mixed=true" in step["body"] for step in regression))
        self.assertTrue(any("ORNITH_CACHE_INTERRUPTION" in step["body"] for step in regression))
        self.assertTrue(any("ORNITH_LOCAL_API realTCP=true" in step["body"] for step in regression))
        self.assertEqual(regression[-1]["name"], "Compile the opt-in Flint app for physical iPhone (unsigned)")
        for step in qpack + regression:
            self.assertNotIn("${{", step["body"])
            self.assertNotIn("xcodebuild archive", step["body"])
            self.assertNotIn("deliver_testflight.py", step["body"])

    def test_ambiguous_range_escape_or_credential_step_is_rejected(self):
        def fixture(step):
            return {"jobs": {"tests": {"steps": [{"name": "first", "run": "true"}, step]}}}
        bad_steps = [
            {"name": "first", "run": "true"},
            {"name": "last", "run": "true", "working-directory": "../outside"},
            {"name": "last", "run": "xcodebuild archive"},
            {"name": "last", "run": "true", "env": {"EXPO_TOKEN": "fixture"}},
            {"name": "last", "uses": "unexpected/action@v1"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / ".github/workflows"
            directory.mkdir(parents=True)
            for step in bad_steps:
                (directory / "fixture.yml").write_text(json.dumps(fixture(step)))
                with self.assertRaises(ValueError):
                    checks.workflow_steps("fixture.yml", "tests", "first", "last", root=root)


if __name__ == "__main__":
    unittest.main()
