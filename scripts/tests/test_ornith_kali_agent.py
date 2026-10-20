import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "ornith_kali_agent.py"
spec = importlib.util.spec_from_file_location("ornith_kali_agent", SOURCE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class KaliAgentTests(unittest.TestCase):
    def scope(self, folder, authorized=True):
        path = Path(folder) / "scope.json"
        path.write_text(json.dumps({"authorized": authorized, "ips": ["192.0.2.10"],
                                    "ports": [443]}), encoding="utf-8")
        return path

    def test_requires_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                agent.parse_scope(self.scope(folder, authorized=False))

    def test_blocks_out_of_scope_and_unsupported_actions(self):
        ips, ports = {"192.0.2.10"}, {443}
        for action in [
            {"action": "tcp_connect", "ip": "192.0.2.11", "port": 443},
            {"action": "tcp_connect", "ip": "192.0.2.10", "port": 22},
            {"action": "shell", "command": "whoami"},
            {"action": "tcp_connect", "ip": "192.0.2.10", "port": 443, "command": "extra"},
        ]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                agent.validate_action(action, ips, ports)

    def test_dry_run_never_connects(self):
        with tempfile.TemporaryDirectory() as folder:
            scope = self.scope(folder)
            report = Path(folder) / "report.json"
            with patch.object(agent, "tcp_connect", side_effect=AssertionError("network access")):
                result = agent.run(scope, "http://127.0.0.1:8080/v1", "", report, dry_run=True)
            self.assertEqual(result["steps"], 0)
            self.assertEqual(result["outcome"]["status"], "finished")

    def test_repeated_action_stops_without_second_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            scope = self.scope(folder)
            action = json.dumps({"action": "tcp_connect", "ip": "192.0.2.10", "port": 443})
            with patch.object(agent, "ask_model", return_value=action), patch.object(
                agent, "tcp_connect", return_value={"status": "open"}
            ) as probe, patch.object(agent.time, "sleep"):
                result = agent.run(scope, "http://127.0.0.1:8080/v1", "test", Path(folder) / "report.json")
            self.assertEqual(result["outcome"]["status"], "repeated_action")
            self.assertEqual(probe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
