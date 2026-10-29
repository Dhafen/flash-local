"""Real tiny Git repos: a moved upstream branch must not change our runtime."""
import pathlib
import importlib.util
import subprocess
import tempfile
import unittest
from unittest.mock import patch
spec = importlib.util.spec_from_file_location("prepare_ornith_les", pathlib.Path(__file__).resolve().parents[1] / "prepare_ornith_les.py")
les = importlib.util.module_from_spec(spec)
spec.loader.exec_module(les)


class LESSourcePinTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.origin = self.root / "origin"
        self.origin.mkdir()
        self.git("init", "--initial-branch", "ios-omni")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.autocrlf", "false")
        (self.origin / "fixture.txt").write_text("reviewed\n")
        self.git("add", "fixture.txt")
        self.git("commit", "-m", "Reviewed fixture")
        self.reviewed = self.git("rev-parse", "HEAD").strip()
        (self.origin / "fixture.txt").write_text("unreviewed branch update\n")
        self.git("commit", "-am", "Advance the branch")
        self.latest = self.git("rev-parse", "HEAD").strip()
        self.build = self.root / "build"
        self.build.mkdir()

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.origin), *arguments], text=True, stderr=subprocess.DEVNULL)

    def run_command(self, arguments):
        subprocess.run([str(x) for x in arguments], cwd=self.build,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_new_source_uses_reviewed_commit_after_branch_advances(self):
        with patch.object(les, "REVISION", self.reviewed), patch.object(les, "REPOSITORY", self.origin.as_uri()), \
             patch.object(les, "patched_files", return_value={"fixture.txt": "reviewed with local patch\n"}):
            source = les.prepare(self.build, self.run_command)
            head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(head, self.reviewed)
            self.assertNotEqual(head, self.latest)
            self.assertEqual((source / "fixture.txt").read_text(), "reviewed with local patch\n")
            self.assertEqual(les.prepare(self.build, self.run_command), source)

    def test_existing_unreviewed_checkout_is_rejected_and_not_reset(self):
        source = self.build / ".work/ornith-les-source"
        source.parent.mkdir()
        self.run_command(["git", "clone", self.origin.as_uri(), source])
        before = (source / "fixture.txt").read_bytes()
        with patch.object(les, "REVISION", self.reviewed):
            with self.assertRaisesRegex(ValueError, "refusing an unreviewed revision"):
                les.prepare(self.build, self.run_command)
        self.assertEqual((source / "fixture.txt").read_bytes(), before)
        self.assertEqual(subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(), self.latest)


if __name__ == "__main__":
    unittest.main()
