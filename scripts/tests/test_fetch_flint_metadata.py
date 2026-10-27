import hashlib
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fetch_flint_metadata import fetch


class FlintMetadataFetchTests(unittest.TestCase):
    def response(self, status, content_range, body=b"abc"):
        response = MagicMock()
        response.status = status
        response.headers = {"Content-Range": content_range}
        response.read.return_value = body
        response.__enter__.return_value = response
        return response

    def test_ignored_range_aborts_before_reading_any_full_model_data(self):
        response = self.response(200, None)
        with tempfile.TemporaryDirectory() as root, patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(ValueError): fetch("url", pathlib.Path(root) / "header", 3, "unused", ranged=True)
            response.read.assert_not_called()

    def test_wrong_total_size_aborts_before_reading(self):
        response = self.response(206, "bytes 0-2/999")
        with tempfile.TemporaryDirectory() as root, patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(ValueError): fetch("url", pathlib.Path(root) / "header", 3, "unused", ranged=True)
            response.read.assert_not_called()

    def test_exact_range_and_hash_can_be_reused_without_network(self):
        response = self.response(206, "bytes 0-2/25992966976")
        with tempfile.TemporaryDirectory() as root, patch("urllib.request.urlopen", return_value=response) as get:
            path = pathlib.Path(root) / "header"
            fetch("url", path, 3, hashlib.sha256(b"abc").hexdigest(), ranged=True)
            self.assertEqual(path.read_bytes(), b"abc")
            response.read.assert_called_once_with(4)
            fetch("url", path, 3, hashlib.sha256(b"abc").hexdigest(), ranged=True)
            self.assertEqual(get.call_count, 1)

    def test_oversized_response_is_not_written(self):
        response = self.response(200, None, b"abcd")
        with tempfile.TemporaryDirectory() as root, patch("urllib.request.urlopen", return_value=response):
            path = pathlib.Path(root) / "header"
            with self.assertRaises(ValueError): fetch("url", path, 3, hashlib.sha256(b"abc").hexdigest())
            self.assertFalse(path.exists())

    def test_changed_cached_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root, patch("urllib.request.urlopen") as get:
            path = pathlib.Path(root) / "header"
            path.write_bytes(b"def")
            with self.assertRaises(ValueError): fetch("url", path, 3, hashlib.sha256(b"abc").hexdigest())
            get.assert_not_called()
            self.assertEqual(path.read_bytes(), b"def")


if __name__ == "__main__": unittest.main()
