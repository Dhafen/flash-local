import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("flint_eas", ROOT / "scripts/run_flint_eas.py")
eas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eas)


class FlintEASTests(unittest.TestCase):
    def test_native_stage_runs_real_builders_and_three_api_formats(self):
        steps = dict(eas.plan("native"))
        self.assertIn("scripts/build_flint_runtime.py", steps["flint-frameworks"])
        self.assertNotIn("--mac-only", steps["flint-frameworks"])
        self.assertEqual(sum(x.startswith("flint-api-") for x in steps), 3)
        self.assertIn("--quantizer", steps["flint-graphs"])

    def test_compile_uses_xcodebuild_and_preserves_original_backend(self):
        steps = dict(eas.plan("compile"))
        self.assertIn("scripts/build_ornith_les.py", steps["les-frameworks"])
        self.assertEqual(steps["flint-swift-tests"][:2], ["xcodebuild", "test"])
        self.assertIn("generic/platform=iOS", steps["iphone-release-compile"])
        self.assertIn("CODE_SIGNING_ALLOWED=NO", steps["iphone-release-compile"])
        for command in steps.values():
            self.assertNotIn("archive", command)
            self.assertNotIn("scripts/download_flint_model.py", command)

    def test_wrong_host_is_rejected_without_claiming_apple_execution(self):
        with patch.object(eas.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "real Apple-silicon"):
                eas.host_check()

    def test_device_compile_and_simulator_controls_precede_host_xctest(self):
        steps = [name for name, _ in eas.plan("visual")]
        self.assertLess(steps.index("iphone-release-compile"), steps.index("apple-test-host"))
        self.assertLess(steps.index("ios-visual-tests"), steps.index("flint-swift-tests"))
        with patch.object(eas.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "ephemeral EAS Mac"):
                eas.prepare_apple_test_host()

    def test_profiles_never_sign_autoincrement_or_use_large_paid_workers(self):
        config = json.loads((ROOT / "eas.json").read_text())
        self.assertEqual(config["cli"]["version"], "27.1.0")
        self.assertTrue(config["cli"]["requireCommit"])
        base = config["build"]["flint-native"]
        self.assertTrue(base["ios"]["withoutCredentials"])
        self.assertFalse(base["ios"]["autoIncrement"])
        self.assertEqual(base["ios"]["resourceClass"], "medium")
        self.assertNotIn("submit", config)
        self.assertEqual(json.loads((ROOT / "app.json").read_text())["expo"]["ios"]["bundleIdentifier"], "com.dhafen.flashlocal")

    def test_artifacts_include_diagnostics_but_exclude_weights_and_private_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            output = root / ".work/eas-output"
            output.mkdir(parents=True)
            (output / "validation.json").write_text('{"status":"fixture"}')
            (output / "credentials.json.p8").write_text("private fixture")
            (output / "model.gguf").write_text("model fixture")
            (root / "credentials.json").write_text("outside fixture")
            with patch.object(eas, "ROOT", root), patch.object(eas, "OUTPUT", output):
                eas.collect()
            with zipfile.ZipFile(output / "reports.zip") as archive:
                self.assertEqual(set(archive.namelist()), {".work/eas-output/validation.json", "index.json"})

    def test_invalid_stage_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            eas.plan("delivery")

    def test_metadata_dependency_does_not_add_an_expo_or_react_app_runtime(self):
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(package.get("dependencies", {}), {})
        self.assertEqual(package["devDependencies"], {"expo-modules-autolinking": "2.1.15"})
        self.assertNotIn("expo", package.get("scripts", {}))


if __name__ == "__main__":
    unittest.main()
