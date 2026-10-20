"""Actual owned-loopback transport plus clearly marked synthetic model replies."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "flash_local_agent.py"
spec = importlib.util.spec_from_file_location("flash_local_agent", SOURCE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)
KEY = "public-test-credential-" * 3


def function_call(ip="127.0.0.1", port=443, call_id="call_public"):
    return {"id": call_id, "type": "function", "function": {"name": "tcp_connect", "arguments": json.dumps({"ip": ip, "port": port})}}


class AgentTests(unittest.TestCase):
    def scope(self, folder, **fields):
        scope = {"authorized": True, "ips": ["127.0.0.1"], "ports": [443], **fields}
        path = Path(folder) / "scope.json"
        path.write_text(json.dumps(scope), encoding="utf-8")
        return path

    def test_authorization_expiry_duplicate_and_target_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            for fields in [{"authorized": False}, {"ips": ["example.com"]}, {"ips": ["0.0.0.0"]},
                           {"ips": ["192.0.2.0/24"]}, {"ports": [True]}, {"ports": [65536]},
                           {"allowed_tools": ["shell"]}, {"expires_at": "2020-01-01T00:00:00Z"}]:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    agent.load_scope(self.scope(folder, **fields))
            for encoded in ['{"a":1,"a":2}', '{"a":NaN}']:
                with self.assertRaises(ValueError):
                    agent.strict_json(encoded)

    def test_validation_cannot_expand_scope_or_coerce_argument_types(self):
        scope = {"ips": ["127.0.0.1"], "ports": [443], "allowed_tools": ["tcp_connect"], "expires_at": None}
        self.assertEqual(agent.validate_call(function_call(), scope)[0], "tcp_connect")
        for call in [function_call(ip="192.0.2.10"), function_call(port=22), function_call(port=True),
                     {**function_call(), "command": "extra"}, {**function_call(), "id": "invalid ID"},
                     {**function_call(), "function": {"name": "shell", "arguments": "{}"}},
                     {**function_call(), "function": {"name": "tcp_connect", "arguments": '{"ip":"127.0.0.1","port":443,"port":22}'}}]:
            with self.subTest(call=call), self.assertRaises(ValueError):
                agent.validate_call(call, scope)

    def test_endpoint_never_sends_credentials_to_remote_or_redirected_hosts(self):
        for endpoint in ["http://example.com/v1", "https://8.8.8.8/v1", "http://127.0.0.1/v1?x=y",
                         "http://user:pass@127.0.0.1/v1", "http://127.0.0.1/other", "ftp://127.0.0.1/v1"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                agent.LocalClient(endpoint, KEY)
        with self.assertRaises(ValueError):
            agent.NoRedirect().redirect_request(None)

    def test_dry_run_and_denied_action_never_execute_network_tools(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(agent, "execute_tool", side_effect=AssertionError("network")):
            scope = self.scope(folder)
            client = unittest.mock.Mock()
            result = agent.run(scope, client, Path(folder) / "dry.jsonl", dry_run=True)
            self.assertEqual(result["outcome"]["status"], "dry_run")
            client.complete.assert_not_called()
            client.complete.side_effect = [({"role": "assistant", "content": None, "tool_calls": [function_call()]}, {}),
                                           ({"role": "assistant", "content": "Denied."}, {})]
            result = agent.run(scope, client, Path(folder) / "deny.jsonl", approval="deny")
            self.assertEqual(result["observed_tool_calls"], 0)
            history = client.complete.call_args.args[0]
            self.assertEqual(history[-1]["role"], "tool")
            self.assertIn("denied_by_operator", history[-1]["content"])

    def test_repetition_out_of_scope_and_cancellation_stop_before_another_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            scope = self.scope(folder)
            client = unittest.mock.Mock()
            call = ({"role": "assistant", "content": None, "tool_calls": [function_call()]}, {})
            client.complete.return_value = call
            with patch.object(agent, "execute_tool", return_value={"observation": "synthetic"}) as execute:
                result = agent.run(scope, client, Path(folder) / "repeat.jsonl", approval="scope")
            self.assertEqual(result["outcome"]["status"], "stopped"); self.assertEqual(execute.call_count, 1)
            client.complete.side_effect = KeyboardInterrupt()
            result = agent.run(scope, client, Path(folder) / "cancel.jsonl", approval="scope")
            self.assertEqual(result["outcome"]["status"], "cancelled"); self.assertEqual(result["observed_tool_calls"], 0)

    def test_compaction_retains_matched_calls_and_bounded_observation_digest(self):
        first = [{"role": "system", "content": "Explicit scope"}, {"role": "user", "content": "Original task"}]
        history = first[:]
        for index in range(5):
            history += [{"role": "assistant", "content": None, "tool_calls": [function_call(call_id=f"call_{index}")]},
                        {"role": "tool", "tool_call_id": f"call_{index}", "content": "x" * 2000}]
        compact = agent.compact_history(history, [{"status": "open"}] * 5, maximum=8000)
        self.assertEqual([x["role"] for x in compact], ["system", "user", "assistant", "tool"])
        self.assertEqual(compact[-1]["tool_call_id"], compact[-2]["tool_calls"][0]["id"])
        self.assertIn("Original task", compact[1]["content"])
        bounded = agent.strict_json(agent.bounded_result({"long": "a" * 10000}, maximum=1024))
        self.assertTrue(bounded["truncated"]); self.assertEqual(len(bounded["sha256"]), 64)

    def test_scoped_http_head_uses_owned_loopback_and_never_follows_redirect(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_HEAD(self):
                seen.append((self.command, self.path, self.headers.get("Authorization")))
                self.send_response(302)
                self.send_header("Location", "http://192.0.2.55/never")
                self.send_header("Server", "owned-fixture")
                self.send_header("Set-Cookie", "excluded")
                self.end_headers()
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            result = agent.execute_tool("http_headers", {"ip": "127.0.0.1", "port": server.server_port, "path": "/owned", "tls": False})
        finally:
            server.shutdown(); server.server_close(); thread.join()
        self.assertEqual(seen, [("HEAD", "/owned", None)])
        self.assertEqual(result["observation"]["status_code"], 302)
        self.assertFalse(result["observation"]["redirect_followed"])
        self.assertNotIn("set-cookie", result["observation"]["headers"])

    def test_real_owned_tcp_api_tool_round_trip_and_private_log(self):
        posts = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.asserted_key = self.headers.get("Authorization")
                if self.asserted_key != "Bearer " + KEY:
                    self.send_error(401); return
                request = agent.strict_json(self.rfile.read(int(self.headers["Content-Length"])))
                posts.append(request)
                # Explicit synthetic inference fixture over real loopback HTTP.
                message = {"role": "assistant", "content": "Observed the owned local listener."} if len(posts) == 2 else {
                    "role": "assistant", "content": None, "tool_calls": [function_call(port=self.server.server_port)]}
                data = json.dumps({"choices": [{"message": message, "finish_reason": "stop" if len(posts) == 2 else "tool_calls"}],
                                   "usage": {"synthetic_fixture": True}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                scope = self.scope(folder, ports=[server.server_port])
                client = agent.LocalClient(f"http://127.0.0.1:{server.server_port}/v1", KEY)
                log = Path(folder) / "session.jsonl"
                result = agent.run(scope, client, log, approval="scope")
                self.assertEqual(result["outcome"]["status"], "finished")
                self.assertEqual(result["observed_tool_calls"], 1)
                self.assertEqual(posts[1]["messages"][-1]["role"], "tool")
                self.assertIn('"open"', posts[1]["messages"][-1]["content"])
                self.assertNotIn(KEY, log.read_text())
                with self.assertRaises(FileExistsError):
                    agent.run(scope, client, log, dry_run=True)
        finally:
            server.shutdown(); server.server_close(); thread.join()
        print("KALI_AGENT_CONTROL realOwnedLoopbackHTTP=true scopedTCP=true structuredToolRoundTrip=true logNoKey=true syntheticProvider=true trainedInference=false phoneReachability=false")


if __name__ == "__main__":
    unittest.main()
