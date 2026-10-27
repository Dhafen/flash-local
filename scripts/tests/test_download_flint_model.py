import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from download_flint_model import fetch


class Response(io.BytesIO):
    def __init__(self, data, start=0, status=206, total=8):
        super().__init__(data); self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{total-1}/{total}", "Content-Length": str(len(data))}
        self.read_called = False
    def read(self, count=-1): self.read_called = True; return super().read(count)


class FlintDownloadTests(unittest.TestCase):
    payload = b"ggufDATA"
    def setUp(self):
        self.model = dict(schemaVersion=1, repository="example/flint", revision="pinned", fileName="fixture.gguf",
                          bytes=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest())
        self.root = tempfile.TemporaryDirectory(); self.addCleanup(self.root.cleanup)
        self.directory = Path(self.root.name)
        self.identity = patch("download_flint_model.identity", return_value=self.model); self.identity.start(); self.addCleanup(self.identity.stop)
        self.disk = patch("download_flint_model.shutil.disk_usage", return_value=type("Disk", (), dict(free=1 << 40))()); self.disk.start(); self.addCleanup(self.disk.stop)
    def partial(self, data, model=None):
        (self.directory / "fixture.gguf.partial").write_bytes(data)
        (self.directory / "flint-download.json").write_text(json.dumps(model or self.model))
    def test_complete_stream_is_hashed_and_published_without_a_second_payload(self):
        with patch("download_flint_model.urllib.request.urlopen", return_value=Response(self.payload)):
            final = fetch(self.directory)
        self.assertEqual(final.read_bytes(), self.payload)
        self.assertFalse((self.directory / "fixture.gguf.partial").exists())
        receipt = json.loads((self.directory / "flint-verified.json").read_text())
        self.assertTrue(receipt["fullFileSHA256Verified"]); self.assertFalse(receipt["trainedInferenceExecuted"])
        self.assertEqual(receipt["fileIdentity"]["bytes"], 8)
    def test_resume_hashes_existing_bytes_and_requests_only_suffix(self):
        self.partial(self.payload[:3])
        with patch("download_flint_model.urllib.request.urlopen", return_value=Response(self.payload[3:], start=3)) as network:
            self.assertEqual(fetch(self.directory).read_bytes(), self.payload)
        self.assertEqual(network.call_args.args[0].get_header("Range"), "bytes=3-7")
    def test_ignored_range_or_wrong_total_aborts_before_reading(self):
        self.partial(self.payload[:3])
        for response in [Response(self.payload, status=200), Response(self.payload[3:], start=3, total=9)]:
            with patch("download_flint_model.urllib.request.urlopen", return_value=response):
                with self.assertRaises(ValueError): fetch(self.directory)
            self.assertFalse(response.read_called)
            self.assertEqual((self.directory / "fixture.gguf.partial").read_bytes(), self.payload[:3])
    def test_dropped_network_resumes_exact_committed_prefix(self):
        with patch("download_flint_model.urllib.request.urlopen", side_effect=[Response(self.payload[:3]), Response(self.payload[3:], start=3)]), patch("download_flint_model.time.sleep"):
            self.assertEqual(fetch(self.directory).read_bytes(), self.payload)
    def test_wrong_sha_never_becomes_an_admitted_final(self):
        with patch("download_flint_model.urllib.request.urlopen", return_value=Response(b"badMODEL")):
            with self.assertRaises(ValueError): fetch(self.directory)
        self.assertFalse((self.directory / "fixture.gguf").exists()); self.assertFalse((self.directory / "flint-verified.json").exists())
    def test_existing_final_is_rehashed_and_not_replaced(self):
        final = self.directory / "fixture.gguf"; final.write_bytes(self.payload)
        with patch("download_flint_model.urllib.request.urlopen") as network:
            self.assertEqual(fetch(self.directory), final)
            network.assert_not_called()
            final.write_bytes(b"badMODEL")
            with self.assertRaises(ValueError): fetch(self.directory)
        self.assertEqual(final.read_bytes(), b"badMODEL")
    def test_unidentified_partial_and_insufficient_disk_fail_before_network(self):
        self.partial(b"g", {**self.model, "revision": "different"})
        with patch("download_flint_model.urllib.request.urlopen") as network:
            with self.assertRaises(ValueError): fetch(self.directory)
            self.partial(b"g")
            with patch("download_flint_model.shutil.disk_usage", return_value=type("Disk", (), dict(free=8))()):
                with self.assertRaises(ValueError): fetch(self.directory)
            network.assert_not_called()


if __name__ == "__main__": unittest.main()
