import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch, Mock
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

    def test_zero_status_or_skipped_suites_cannot_replace_real_test_execution(self):
        with self.assertRaises(RuntimeError):
            eas.verify_step_log("flint-swift-tests", "** TEST SUCCEEDED **\nExecuted 0 tests")
        suites = ("FlintPolicyTests", "FlintStreamTests", "LocalFlintProtocolTests", "LocalModelAPITests")
        rounds = ("ORNITH_LOCAL_API realTCP=true auth=true json=true sse=true reasoningSeparated=true\n"
                  "LOCAL_API_TOOL_ROUNDTRIP realTCP=true structuredArguments=true matchedToolResult=true\n")
        summary_only = rounds + "\n".join("Test Suite '" + suite + "' passed" for suite in suites)
        with self.assertRaises(RuntimeError):
            eas.verify_step_log("flint-swift-tests", summary_only)
        complete = summary_only + "\n" + "\n".join(
            "Test Case '-[Tests." + suite + " testRequiredBehavior]' passed (0.1 seconds)." for suite in suites)
        eas.verify_step_log("flint-swift-tests", complete)
        with self.assertRaises(RuntimeError):
            eas.verify_step_log("flint-swift-tests", complete.replace("LOCAL_API_TOOL_ROUNDTRIP", "MISSING"))

    def test_gui_transition_drops_root_and_excludes_unrelated_credentials(self):
        command = ["xcodebuild", "test", "-scheme", "minirun-Package"]
        env = {"FLASH_LOCAL_ENABLE_FLINT": "1", "PATH": "/fixture/bin",
               "EXPO_TOKEN": "unrelated-fixture", "APP_STORE_API_KEY": "unrelated-fixture"}
        wrapped = eas.gui_command(command, env, 501)
        self.assertEqual(wrapped[:5], ["sudo", "-n", "/bin/launchctl", "asuser", "501"])
        self.assertIn("#501", wrapped)
        self.assertIn("-H", wrapped)
        self.assertEqual(wrapped[wrapped.index("/usr/bin/env") + 1], "-i")
        self.assertEqual(wrapped[-len(command):], command)
        self.assertNotIn("unrelated-fixture", " ".join(wrapped))
        self.assertIn("FLASH_LOCAL_ENABLE_FLINT=1", wrapped)
        with self.assertRaises(RuntimeError):
            eas.gui_command(command, env, 0)
        with patch.object(eas.subprocess, "run", return_value=Mock(returncode=0, stdout="System\n", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "Aqua session"):
                eas.check_gui_context(env, 501)
        with patch.object(eas.subprocess, "run", return_value=Mock(returncode=0, stdout="Aqua\n", stderr="")):
            self.assertEqual(eas.check_gui_context(env, 501)["manager"], "Aqua")

    def test_windows_archive_keeps_exact_published_notice_bytes(self):
        notices = (ROOT / "Apps/Minirun/Resources/OpenSourceNotices.txt").read_bytes()
        marker = b"Ornith 1.5 publisher license (verbatim)\n\n"
        self.assertEqual(notices.count(marker), 1)
        self.assertEqual(hashlib.sha256(notices.split(marker)[1]).hexdigest(),
                         "ba0b4cc44cd824e63b2aa20691ab8707ed7407a91b3772d8ed47896d8c4d9629")
        self.assertEqual(hashlib.sha256((ROOT / "Apps/Minirun/Resources/QwenModelLicense.txt").read_bytes()).hexdigest(),
                         "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08")

    def test_windows_source_archive_preserves_apple_audit_line_boundaries(self):
        # CRLF is one Swift Character, so the existing source reader cannot
        # find a standalone newline after // comments in a CRLF archive.
        for name in ("RootView.swift", "ConversationView.swift", "DestinationPicker.swift"):
            path = ROOT / "Apps/Minirun/Sources/Screens" / name
            self.assertNotIn(b"\r", path.read_bytes(), name)
        destination = (ROOT / "Apps/Minirun/Sources/Screens/DestinationPicker.swift").read_bytes()
        self.assertIn(b"#if os(macOS)\n            .frame(minWidth: 560", destination)

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
