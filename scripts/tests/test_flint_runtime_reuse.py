import hashlib
import importlib.util
import json
from pathlib import Path
import plistlib
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location("flint_runtime_builder", ROOT / "scripts/build_flint_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
finally:
    sys.path.pop(0)


class FlintImmutableRuntimeTests(unittest.TestCase):
    def fixture(self, root):
        source, output, native = (root / name for name in ("source", "output", "native"))
        for path in (source, output, native): path.mkdir()
        (native / "FlashFlint.h").write_bytes(b"int fixture(void);\n")
        (native / "exports.txt").write_text("_flf_fixture\n")
        source_record = {"baseRevision": "fixture-only-no-code-built"}
        (source / "flash-local-flint-source.json").write_text(json.dumps(source_record))
        framework = output / "FlashFlint.xcframework"
        libraries, slices = [], []
        for name in ("macos", "ios", "ios-simulator"):
            library = framework / name / "FlashFlint.framework"
            (library / "Headers").mkdir(parents=True)
            (library / "Modules").mkdir()
            binary = b"unit-fixture-not-a-real-framework:" + name.encode()
            (library / "FlashFlint").write_bytes(binary)
            (library / "Headers/FlashFlint.h").write_bytes((native / "FlashFlint.h").read_bytes())
            (library / "Modules/module.modulemap").write_text('framework module FlashFlint {\n  umbrella header "FlashFlint.h"\n  export *\n}\n')
            item = {"SupportedPlatform": "ios" if name == "ios-simulator" else name,
                    "LibraryIdentifier": name, "LibraryPath": "FlashFlint.framework", "SupportedArchitectures": ["arm64"]}
            if name == "ios-simulator": item["SupportedPlatformVariant"] = "simulator"
            libraries.append(item)
            slices.append({"platform": name, "sha256": hashlib.sha256(binary).hexdigest(),
                           "moduleImportVerified": True, "symbolIsolated": True, "exportedSymbols": ["_flf_fixture"]})
        (framework / "Info.plist").write_bytes(plistlib.dumps({"AvailableLibraries": libraries}))
        record = {"source": source_record, "nativeFiles": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native.iterdir()},
                  "slices": slices, "trainedInferenceVerified": False, "iPhoneVerified": False}
        (output / "build-manifest.json").write_text(json.dumps(record))
        return source, output, native

    def test_complete_existing_output_is_verified_without_being_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, native = self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(runtime.verify_existing(source, output, native), 3)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_binary_header_module_and_input_changes_are_rejected(self):
        for changed in ("binary", "header", "module", "input", "source", "missing-platform"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                source, output, native = self.fixture(Path(tmp))
                library = output / "FlashFlint.xcframework/ios/FlashFlint.framework"
                if changed == "binary": (library / "FlashFlint").write_bytes(b"changed")
                elif changed == "header": (library / "Headers/FlashFlint.h").write_bytes(b"changed")
                elif changed == "module": (library / "Modules/module.modulemap").write_text("changed")
                elif changed == "input": (native / "FlashFlint.h").write_bytes(b"changed")
                elif changed == "source": (source / "flash-local-flint-source.json").write_text("{}")
                else:
                    path = output / "build-manifest.json"
                    record = json.loads(path.read_text()); record["slices"].pop(); path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    runtime.verify_existing(source, output, native)


if __name__ == "__main__": unittest.main()
