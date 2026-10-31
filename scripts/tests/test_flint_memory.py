import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flint_memory import plan


class FlintMemoryTests(unittest.TestCase):
    def test_compact_attention_8k_has_a_bounded_expert_bank(self):
        p = plan(8192)
        self.assertTrue(p["admitted"])
        self.assertEqual(p["expertSlots"], 24)
        self.assertEqual(p["expertBankBytes"], 47_185_920)
        self.assertEqual(p["paddedKVBytes"], 289_931_264)
        self.assertLess(p["estimatedProcessBytes"], 5_500_000_000)
        self.assertFalse(p["iPhoneStabilityVerified"])

    def test_65k_is_not_admitted_by_overriding_the_real_window(self):
        p = plan(65536)
        self.assertFalse(p["admitted"])
        self.assertEqual(p["paddedKVBytes"], 1_934_098_432)
        self.assertGreater(p["estimatedProcessBytes"], p["effectiveCeilingBytes"])

    def test_observed_available_memory_restricts_admission(self):
        self.assertFalse(plan(8192, available_bytes=2_000_000_000)["admitted"])
        self.assertTrue(plan(32768)["admitted"])
        self.assertTrue(plan(32768)["estimated"])

    def test_bank_capacity_saturates_at_actual_expert_count(self):
        self.assertEqual(plan(8192, physical_batch=256)["expertSlots"], 256)
        self.assertEqual(plan(8192, physical_batch=128)["expertBankBytes"], plan(8192, physical_batch=256)["expertBankBytes"])

    def test_opt_in_expert_cache_is_separately_budgeted(self):
        default = plan(8192, physical_batch=1)
        cache = plan(8192, physical_batch=1, expert_cache_slots=128)
        self.assertEqual(default["expertSlots"], 6)
        self.assertEqual(cache["expertSlots"], 128)
        self.assertEqual(cache["requestedExpertCacheSlots"], 128)
        self.assertGreater(cache["expertBankBytes"], default["expertBankBytes"])
        self.assertEqual(cache["estimatedProcessBytes"] - default["estimatedProcessBytes"],
                         cache["expertBankBytes"] - default["expertBankBytes"])
        self.assertFalse(cache["iPhoneStabilityVerified"])
        self.assertFalse(plan(8192, physical_batch=1, expert_cache_slots=128,
                              process_ceiling=default["estimatedProcessBytes"])["admitted"])

    def test_cache_slot_bounds_and_saturation(self):
        self.assertEqual(plan(8192, physical_batch=1, expert_cache_slots=256)["expertSlots"], 256)
        self.assertEqual(plan(8192, physical_batch=1, expert_cache_slots=0)["expertSlots"], 6)
        for slots in (-1, 257, True, 1.5):
            with self.assertRaises(ValueError): plan(8192, expert_cache_slots=slots)

    def test_layer_resident_cache_is_additive_and_budgeted(self):
        baseline = plan(8192, physical_batch=1)
        window = plan(8192, physical_batch=1, decode_cache_window=32)
        self.assertEqual(window["decodeCacheWindow"], 32)
        self.assertEqual(window["decodeCacheBytes"], 47 * 32 * 1_966_080)
        self.assertEqual(window["estimatedProcessBytes"] - baseline["estimatedProcessBytes"],
                         window["decodeCacheBytes"])
        self.assertFalse(plan(8192, physical_batch=1, decode_cache_window=32,
                              process_ceiling=baseline["estimatedProcessBytes"])["admitted"])
        self.assertEqual(plan(8192)["decodeCacheBytes"], 0)
        for value in (-1, 65, True, 1.5):
            with self.assertRaises(ValueError):
                plan(8192, decode_cache_window=value)

    def test_invalid_geometry_and_budgets_rejected(self):
        for context in [0, -1, 262145, True, 1.5]:
            with self.assertRaises(ValueError): plan(context)
        for batch in [0, -1, 257, True, 1.5]:
            with self.assertRaises(ValueError): plan(8192, physical_batch=batch)
        with self.assertRaises(ValueError): plan(8192, available_bytes=-1)
        with self.assertRaises(ValueError): plan(8192, process_ceiling=-1)


if __name__ == "__main__": unittest.main()
