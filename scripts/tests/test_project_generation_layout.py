import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("project_generation", ROOT / "scripts/generate_xcode_project.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


class ProjectGenerationLayoutTests(unittest.TestCase):
    def test_canonical_tree_contains_only_declared_source_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "build"
            for name in ("Sources", "Tests", "Resources"):
                directory = root / "Apps/Minirun" / name
                directory.mkdir(parents=True)
                (directory / "fixture.txt").write_text(name)
            for name in ("project.yml", "Info-iOS.plist", "Info-macOS.plist", "Minirun-iOS.entitlements", "Minirun-macOS.entitlements"):
                (root / "Apps/Minirun" / name).write_text(name)
            (root / "Package.swift").write_text("root fixture")
            experiment = root / "Experiments/OrnithPerformance"
            experiment.mkdir(parents=True)
            (experiment / "Package.swift").write_text("experiment fixture")
            (root / "model.gguf").write_text("not a real model; must never enter the generation tree")
            (root / "credentials.p8").write_text("not a credential; must never be copied")
            app = generation.generation_tree(root, parent / "temporary")
            mirror = app.parents[1]
            self.assertEqual(mirror.name, "flash-local")
            self.assertEqual((app / "Sources/fixture.txt").read_text(), "Sources")
            self.assertEqual((mirror / "Package.swift").read_text(), "root fixture")
            self.assertFalse(list(mirror.rglob("*.gguf")))
            self.assertFalse(list(mirror.rglob("*.p8")))
            self.assertFalse((mirror / ".git").exists())
            self.assertTrue((root / "model.gguf").exists())


if __name__ == "__main__":
    unittest.main()
