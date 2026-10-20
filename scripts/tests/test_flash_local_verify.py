import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_local_verify import benchmark, stream_completion, tool_round_trip


class VerificationHarnessTests(unittest.TestCase):
    def chunks(self, done=True, usage=True):
        def event(**values):
            return "data: " + json.dumps({"id": "chatcmpl_fixture", **values}) + "\n\n"
        text = event(choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        text += event(choices=[{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_test", "type": "function",
            "function": {"name": "local_", "arguments": '{"a":'}}]}, "finish_reason": None}])
        text += event(choices=[{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "sum", "arguments": '50,"b":5}'}}]}, "finish_reason": None}])
        text += event(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
        if usage:
            text += event(choices=[], usage={"completion_tokens": 30})
        if done:
            text += "data: [DONE]\n\n"
        return text.encode()

    def test_reassembles_valid_partial_function_arguments_and_requires_complete_sse(self):
        client = Mock()
        client.request.return_value = io.BytesIO(self.chunks())
        result = stream_completion(client, {})
        call = result["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"], {"name": "local_sum", "arguments": '{"a":50,"b":5}'})
        for chunks in [self.chunks(done=False), self.chunks(usage=False), b'data: {"error":{"message":"fixture"}}\n\n']:
            client.request.return_value = io.BytesIO(chunks)
            with self.assertRaises(ValueError):
                stream_completion(client, {})

    def test_tool_receipt_is_created_after_call_and_read_from_matching_result(self):
        client = Mock()
        seen = []
        def respond(path, payload):
            seen.append(payload)
            if len(seen) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_test", "type": "function",
                    "function": {"name": "local_sum", "arguments": '{"a":50,"b":5}'}}]}
                return {"choices": [{"message": message, "finish_reason": "tool_calls"}], "usage": {}}
            result = json.loads(payload["messages"][-1]["content"])
            self.assertEqual(result["sum"], 55)
            self.assertEqual(payload["messages"][-1]["tool_call_id"], "call_test")
            return {"choices": [{"message": {"role": "assistant", "content": result["receipt"]}, "finish_reason": "stop"}], "usage": {}}
        client.json.side_effect = respond
        result = tool_round_trip(client, "fixture")
        self.assertTrue(result["tool_result_receipt_observed"])
        self.assertNotIn("LOCAL-", seen[0]["messages"][0]["content"])

    def test_invalid_generated_argument_never_reaches_a_tool_result(self):
        client = Mock()
        client.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
            {"id": "call_test", "function": {"name": "local_sum", "arguments": '{"a":true,"b":5}'}}]}}]}
        with self.assertRaises(ValueError):
            tool_round_trip(client, "fixture")
        self.assertEqual(client.json.call_count, 1)

    def test_benchmark_never_invents_native_metrics_or_a_cross_model_comparison(self):
        client = Mock()
        client.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "fixture"}}],
                                    "usage": {"completion_tokens": 4}}
        results = benchmark(client, "fixture")
        self.assertEqual(len(results), 6)
        self.assertEqual({row["configuration"] for row in results}, {"off", "small", "medium"})
        for row in results:
            self.assertFalse(row["minimum_128_tokens_met"])
            self.assertTrue(row["same_model_comparison"])
            self.assertFalse(row["os_cache_coldness_controlled"])
            self.assertNotIn("decode_tokens_per_second", row)


if __name__ == "__main__":
    unittest.main()
