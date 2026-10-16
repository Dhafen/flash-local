import importlib.util
import json
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "inspect_ornith_mlx.py"
SPEC = importlib.util.spec_from_file_location("ornith_mlx_metadata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OrnithMLXMetadataTests(unittest.TestCase):
    def header(self, **changes):
        tensor = dict(dtype="U32", shape=[2, 8], data_offsets=[0, 64])
        tensor.update(changes)
        return json.dumps({"model.weight": tensor}, sort_keys=True).encode()

    def test_exact_tensor_shape_and_source_length(self):
        data = self.header()
        self.assertIn("model.weight", MODULE.checked_header(data, 8 + len(data) + 64))

    def test_unknown_quantized_storage_is_refused(self):
        for dtype in ["F4", "I4", "MXFP4", "OBJECT"]:
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                data = self.header(dtype=dtype)
                MODULE.checked_header(data, 8 + len(data) + 64)

    def test_shape_cannot_disagree_with_bytes(self):
        for shape in [[], [0, 8], [-1, 8], [2, 9], [True, 16]]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                data = self.header(shape=shape)
                MODULE.checked_header(data, 8 + len(data) + 64)

    def test_gaps_overlaps_and_trailing_payload_are_refused(self):
        for offsets, extra in [([1, 65], 1), ([0, 63], 0), ([0, 64], 1)]:
            with self.subTest(offsets=offsets, extra=extra), self.assertRaises(ValueError):
                data = self.header(data_offsets=offsets)
                MODULE.checked_header(data, 8 + len(data) + 64 + extra)


if __name__ == "__main__":
    unittest.main()
