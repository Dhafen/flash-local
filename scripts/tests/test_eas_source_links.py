"""Actual Git index symlinks, including Windows's text-file representation."""
import importlib.util
import os
import pathlib
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("flint_eas_links", pathlib.Path(__file__).resolve().parents[1] / "run_flint_eas.py")
eas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eas)


class EASSourceLinksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.sandbox = pathlib.Path(temporary.name)
        self.root = self.sandbox / "repo"
        self.root.mkdir()
        self.git("init")
        self.git("config", "core.symlinks", "false")
        self.git("config", "core.autocrlf", "false")
        (self.root / "source.swift").write_bytes(b"// actual source fixture\n")
        self.git("add", "source.swift")

    def git(self, *args, data=None):
        return subprocess.run(["git", "-C", str(self.root), *args], input=data,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout

    def link(self, target):
        data = target.encode()
        oid = self.git("hash-object", "-w", "--stdin", data=data).decode().strip()
        self.git("update-index", "--add", "--cacheinfo", f"120000,{oid},link.swift")
        (self.root / "link.swift").write_bytes(data)

    def test_recognizes_windows_archive_without_modifying_source(self):
        self.link("source.swift")
        self.assertEqual(eas.git_symlink_plan(self.root), [("link.swift", "source.swift")])
        self.assertEqual((self.root / "link.swift").read_bytes(), b"source.swift")
        self.assertFalse((self.root / "link.swift").is_symlink())
        (self.root / "link.swift").write_text("modified user content")
        with self.assertRaisesRegex(RuntimeError, "modified build link content"):
            eas.git_symlink_plan(self.root)
        self.assertEqual((self.root / "link.swift").read_text(), "modified user content")

    def test_rejects_escape_before_replacing_any_file(self):
        (self.sandbox / "outside.swift").write_bytes(b"keep outside untouched")
        self.link("../outside.swift")
        with self.assertRaisesRegex(RuntimeError, "inside source"):
            eas.git_symlink_plan(self.root)
        self.assertEqual((self.root / "link.swift").read_bytes(), b"../outside.swift")
        self.assertEqual((self.sandbox / "outside.swift").read_bytes(), b"keep outside untouched")

    @unittest.skipIf(os.name == "nt", "Real symlink creation requires a Unix build host")
    def test_restores_real_git_symlink_and_keeps_target_bytes(self):
        self.link("source.swift")
        original = (self.root / "source.swift").read_bytes()
        self.assertEqual(eas.restore_git_symlinks(self.root), ["link.swift"])
        self.assertTrue((self.root / "link.swift").is_symlink())
        self.assertEqual((self.root / "link.swift").read_bytes(), original)
        self.assertEqual((self.root / "source.swift").read_bytes(), original)
