import hashlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy is required for the optional weight replay")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from replay_kinetix_weight_edit import apply_edits, parse_prompt_weights, read_pinned


class KinetixWeightEditTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.base = {"embed.W": np.array([[-0.0, 2.0], [3.0, 4.0]], dtype="<f8")}

    def test_published_edits_apply_in_order_and_undo_byte_for_byte(self):
        candidate, undo, count = apply_edits(
            self.base,
            "embed.W[:, 0] = 9\nembed.W[1, 0] -= 5\nembed.W[0, 1] += 7",
        )
        self.assertEqual(count, 3)
        self.assertEqual(candidate["embed.W"].tolist(), [[9.0, 9.0], [4.0, 4.0]])
        self.assertEqual(self.base["embed.W"].tobytes(),
                         bytes.fromhex("0000000000000080000000000000004000000000000008400000000000001040"))
        self.assertEqual(undo(candidate)["embed.W"].tobytes(), self.base["embed.W"].tobytes())
        candidate["embed.W"][1, 1] = 500
        with self.assertRaises(ValueError):
            undo(candidate)

    def test_invalid_statements_cannot_change_the_base(self):
        before = self.base["embed.W"].tobytes()
        for bad in ("embed.W[0, 2] = 0", "embed.W[:, 0:3] = 0", "embed.W[0, 0] = 1e7",
                    "embed.W[0, 0] = __import__('os')", "embed.W[0] = 0",
                    "embed.W[0, :] = [1]", "embed.W[0, :] = [1, 2, 3]"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                apply_edits(self.base, "embed.W[0, 1] = 10\n" + bad)
            self.assertEqual(self.base["embed.W"].tobytes(), before)

    def test_prompt_parser_restores_tensor_scaling_and_rejects_missing_rows(self):
        prompt = (
            "ignored\nTHE WEIGHTS (complete actor).\n\n"
            "embed.W (2 x 2), scale 1e-2: one line per row (row i)\n"
            "0: 10 -20\n1: 30 40\n\nembed.b (2), scale 1e-3:\n2 -3\n"
        )
        shapes = {"embed.W": (2, 2), "embed.b": (2,)}
        decoded = parse_prompt_weights(prompt, shapes)
        self.assertEqual(decoded["embed.W"].tolist(), [[0.1, -0.2], [0.3, 0.4]])
        self.assertEqual(decoded["embed.b"].tolist(), [0.002, -0.003])
        with self.assertRaises(ValueError):
            parse_prompt_weights(prompt.replace("1: 30 40", "0: 30 40"), shapes)

    def test_pin_rejects_different_bytes_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "source"
            path.write_bytes(b"altered")
            expected = hashlib.sha256(b"original").hexdigest()
            with self.assertRaises(ValueError):
                read_pinned(path, expected)


if __name__ == "__main__":
    unittest.main()
