import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_ream_metadata as fetch


class CachedREAMHeaderTests(unittest.TestCase):
    def test_only_exact_pinned_cache_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ream.header'
            self.assertFalse(fetch.verified_cached_header(path))
            original = b'synthetic metadata only'
            sha = hashlib.sha256(original).hexdigest()
            with patch.object(fetch, 'HEADER_BYTES', len(original)), patch.object(fetch, 'HEADER_SHA256', sha):
                path.write_bytes(original)
                self.assertTrue(fetch.verified_cached_header(path))
                path.write_bytes(original[:-1])
                with self.assertRaises(RuntimeError): fetch.verified_cached_header(path)
                path.write_bytes(b'x' * len(original))
                with self.assertRaises(RuntimeError): fetch.verified_cached_header(path)

    def test_symlink_cache_is_refused_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root / 'target'; target.write_bytes(b'metadata')
            link = root / 'ream.header'
            try: link.symlink_to(target)
            except (OSError, NotImplementedError): self.skipTest('Symlinks unavailable on this host')
            with self.assertRaises(RuntimeError): fetch.verified_cached_header(link)


if __name__ == '__main__': unittest.main()
