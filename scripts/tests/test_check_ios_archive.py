import importlib.util
import pathlib
import plistlib
import struct
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("archive_check", ROOT / "scripts/check_ios_archive.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class ArchiveChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.archive = pathlib.Path(self.folder.name)
        self.app = self.archive / "Products/Applications/FlashLocal.app"
        self.app.mkdir(parents=True)
        self.info = {"CFBundleIdentifier": "com.dhafen.flashlocal", "CFBundleExecutable": "FlashLocal",
                     "DTPlatformName": "iphoneos", "UIFileSharingEnabled": True,
                     "LSSupportsOpeningDocumentsInPlace": True, "UIDeviceFamily": [1]}
        self.write_info()
        self.write_binary(2)
        (self.app / "QwenModelLicense.txt").write_bytes(
            (ROOT / "Apps/Minirun/Resources/QwenModelLicense.txt").read_bytes())

    def write_info(self):
        (self.app / "Info.plist").write_bytes(plistlib.dumps(self.info))

    def write_binary(self, platform):
        header = struct.pack("<8I", 0xFEEDFACF, 0x0100000C, 0, 2, 1, 24, 0, 0)
        command = struct.pack("<6I", 0x32, 24, platform, 18 << 16, 26 << 16, 0)
        (self.app / "FlashLocal").write_bytes(header + command)

    def test_device_archive_reports_only_proven_scope(self):
        result = check.inspect(self.archive)
        self.assertEqual(result["platform"], "iphoneos")
        self.assertFalse(result["signed"])
        self.assertFalse(result["uploadedToTestFlight"])

    def test_simulator_binary_refused_even_when_info_claims_device(self):
        self.write_binary(7)
        with self.assertRaisesRegex(ValueError, "device iOS"):
            check.inspect(self.archive)

    def test_weights_and_keys_never_belong_in_app_bundle(self):
        for suffix in (".gguf", ".safetensors", ".p8", ".p12"):
            with self.subTest(suffix=suffix):
                file = self.app / ("external" + suffix)
                file.write_bytes(b"fixture")
                with self.assertRaisesRegex(ValueError, "Model or signing material"):
                    check.inspect(self.archive)
                file.unlink()

    def test_wrong_identity_and_missing_files_access_refused(self):
        for key, value in (("CFBundleIdentifier", "com.example.other"), ("UIFileSharingEnabled", False)):
            original = self.info[key]
            self.info[key] = value
            self.write_info()
            with self.assertRaisesRegex(ValueError, key):
                check.inspect(self.archive)
            self.info[key] = original

    def test_truncated_commands_and_modified_license_refused(self):
        binary = self.app / "FlashLocal"
        binary.write_bytes(binary.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            check.inspect(self.archive)
        self.write_binary(2)
        (self.app / "QwenModelLicense.txt").write_text("incomplete notice")
        with self.assertRaisesRegex(ValueError, "license"):
            check.inspect(self.archive)


if __name__ == "__main__":
    unittest.main()
