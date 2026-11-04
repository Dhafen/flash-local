import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from xml.etree import ElementTree


SCRIPT = Path(__file__).resolve().parents[1] / "replay_ornith_routing.py"


def fixture(layers, k=1, experts=4):
    return dict(num_experts=experts, experts_per_token=k,
                tokens=len(next(iter(layers.values()))), layers=layers)


class RoutingReplayTests(unittest.TestCase):
    def run_trace(self, data, *args):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "PRIVATE_MARKER.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return subprocess.run([sys.executable, str(SCRIPT), "--trace", str(path), *args],
                                  capture_output=True, text=True)

    def test_b_charges_prefetch_and_misses_without_changing_routes(self):
        proc = self.run_trace(fixture({"0": [[0], [1], [0], [1], [2]]}),
                              "--slots", "1", "--prefetch", "1", "--hold", "2", "--expert-bytes", "10")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        a, b, c = report["strategies"]
        self.assertEqual((a["demand_read_bytes"], a["total_read_bytes"]), (50, 50))
        self.assertEqual((b["demand_read_bytes"], b["prefetch_read_bytes"], b["total_read_bytes"]), (40, 20, 60))
        self.assertEqual((b["useful_prefetch_bytes"], b["unused_prefetch_bytes"]), (10, 10))
        self.assertEqual(b["route_digest"], a["route_digest"])
        self.assertEqual(c["total_read_bytes"], 20)
        self.assertEqual(c["replaced_expert_selections"], 2)
        self.assertEqual(c["changed_route_rows"], 2)
        self.assertNotEqual(c["route_digest"], a["route_digest"])
        self.assertFalse(report["quality_evaluated"])
        self.assertEqual(report["layers"][0]["reuse_token_gap_counts"], {"2": 2})
        self.assertEqual(report["layers"][0]["adjacent_overlap"], 0)

    def test_resident_selected_expert_survives_batch_miss_with_per_layer_bytes(self):
        data = fixture({"0": [[0, 1], [2, 0]], "3": [[0, 1], [1, 0]]}, k=2)
        data["expert_bytes_by_layer"] = {"0": 10, "3": 100}
        proc = self.run_trace(data, "--slots", "2", "--prefetch", "0", "--hold", "1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        a, b, c = report["strategies"]
        self.assertEqual(report["expert_cache_budget_bytes"], 220)
        self.assertEqual(a["total_read_bytes"], 230)
        self.assertEqual(a["demand_hits"], 3)
        self.assertEqual(b["total_read_bytes"], 230)
        self.assertEqual(c["route_digest"], a["route_digest"])
        self.assertEqual([layer["adjacent_overlap"] for layer in report["layers"]], [0.5, 1.0])

    def test_invalid_shape_and_unsafe_capacity_fail_without_echoing_trace(self):
        valid = fixture({"0": [[0, 1], [1, 2]]}, k=2)
        invalid_cases = [
            (fixture({"0": [[0, 0]]}, k=2), ("--slots", "2", "--expert-bytes", "10")),
            (fixture({"0": [[True]]}), ("--slots", "1", "--expert-bytes", "10")),
            ({**valid, "layers": {"PRIVATE_MARKER": [[0, 1], [1, 2]]}},
             ("--slots", "2", "--expert-bytes", "10")),
            (valid, ("--slots", "1", "--expert-bytes", "10")),
            (valid, ("--slots", "2")),
            ({"routingDigest": 123, "routingDecisions": 16}, ("--slots", "2", "--expert-bytes", "10")),
        ]
        for data, args in invalid_cases:
            with self.subTest(args=args, data=data):
                proc = self.run_trace(data, *args)
                self.assertEqual(proc.returncode, 2)
                self.assertNotIn("PRIVATE_MARKER", proc.stdout + proc.stderr)
                self.assertEqual(proc.stdout, "")

    def test_demo_json_and_svg_are_deterministic_and_labeled_synthetic(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "chart.svg"
            args = [sys.executable, str(SCRIPT), "--demo", "sticky", "--svg", str(svg)]
            first = subprocess.run(args, check=True, capture_output=True)
            chart = svg.read_bytes()
            second = subprocess.run(args, check=True, capture_output=True)
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(chart, svg.read_bytes())
            report = json.loads(first.stdout)
            self.assertEqual(report["trace_source"], "synthetic")
            self.assertEqual((report["trace_tokens"], report["num_experts"], report["top_k"]), (128, 256, 8))
            self.assertEqual(len(report["layers"]), 40)
            root = ElementTree.fromstring(chart)
            self.assertIn("Synthetic", "".join(root.itertext()))
            self.assertIn("Adjacent top-k overlap", "".join(root.itertext()))


if __name__ == "__main__":
    unittest.main()
