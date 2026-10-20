#!/usr/bin/env python3
"""Measure a live Flash Local server; never downloads or simulates a model.

The tool test only adds two integers on this computer. It scans no target.
Use --benchmark for three reasoning configurations of the selected model;
that is not a comparison between three different models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import uuid

from flash_local_agent import LocalClient, MODEL, strict_json


def stream_completion(client, payload):
    message = {"role": "assistant", "content": ""}
    calls, completion_id, finish, usage, done = {}, None, None, None, False
    total = 0
    with client.request("/chat/completions", {**payload, "stream": True, "stream_options": {"include_usage": True}}) as response:
        for raw in response:
            total += len(raw)
            if len(raw) > 1 << 20 or total > 4 << 20:
                raise ValueError("Oversized SSE response")
            line = raw.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data: "):
                raise ValueError("Unexpected SSE framing")
            value = line[6:]
            if value == "[DONE]":
                done = True; break
            event = strict_json(value)
            if "error" in event:
                raise ValueError("Server returned a streamed error")
            if completion_id is None:
                completion_id = event.get("id")
            if not completion_id or event.get("id") != completion_id:
                raise ValueError("Completion ID changed during streaming")
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("index") != 0:
                    raise ValueError("Unexpected streaming choice")
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta", {})
                for key in ["content", "reasoning_content"]:
                    if delta.get(key) is not None:
                        if not isinstance(delta[key], str):
                            raise ValueError("Malformed streamed text")
                        message[key] = message.get(key, "") + delta[key]
                for call in delta.get("tool_calls", []):
                    index = call.get("index")
                    if type(index) is not int or not 0 <= index < 16:
                        raise ValueError("Invalid tool-call index")
                    target = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if call.get("id"):
                        target["id"] += call["id"]
                    if call.get("type", "function") != "function":
                        raise ValueError("Unsupported streamed tool type")
                    for key in ["name", "arguments"]:
                        if call.get("function", {}).get(key) is not None:
                            target["function"][key] += call["function"][key]
    if not done or finish is None or usage is None:
        raise ValueError("Stream ended without a finish reason, usage or DONE")
    if calls:
        if sorted(calls) != list(range(len(calls))) or finish != "tool_calls":
            raise ValueError("Incomplete streamed tool calls")
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


def tool_round_trip(client, model, streaming=False):
    receipt = "LOCAL-" + uuid.uuid4().hex
    tool = {"type": "function", "function": {"name": "local_sum", "description": "Add two integers on the operator's computer and return a receipt.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer", "enum": [50]}, "b": {"type": "integer", "enum": [5]}},
                       "required": ["a", "b"], "additionalProperties": False}, "strict": True}}
    user = {"role": "user", "content": "Call local_sum with a=50 and b=5. Then read its tool result and reply with only its receipt string."}
    payload = {"model": model, "messages": [user], "tools": [tool], "tool_choice": "required", "parallel_tool_calls": False,
               "temperature": 0, "enable_thinking": False, "max_tokens": 256}
    complete = (lambda value: stream_completion(client, value)) if streaming else (lambda value: client.json("/chat/completions", value))
    first = complete(payload)
    choice = first["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls", [])
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        raise ValueError("The real model did not supply one typed function call")
    call = calls[0]
    function = call["function"]
    arguments = strict_json(function["arguments"])
    if (function["name"] != "local_sum" or not isinstance(arguments, dict) or set(arguments) != {"a", "b"} or
            type(arguments["a"]) is not int or type(arguments["b"]) is not int or arguments != {"a": 50, "b": 5}):
        raise ValueError("The real model's tool arguments differ from the declared test")
    # This receipt is created only after the model selected its tool. It cannot
    # be answered from the first prompt or from a fixed synthetic completion.
    result = {"sum": arguments["a"] + arguments["b"], "receipt": receipt}
    payload["messages"] = [user, message, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}]
    payload["tool_choice"] = "none"
    followup = complete(payload)
    final = followup["choices"][0]
    observed = isinstance(final["message"].get("content"), str) and receipt in final["message"]["content"]
    return {"streaming": streaming, "structured_call_valid": True, "tool_result_receipt_observed": observed,
            "finish_reason": final.get("finish_reason"), "first_usage": first.get("usage"), "followup_usage": followup.get("usage")}


def benchmark(client, model):
    prompt = "Write 150 numbered, concrete review checks for a Python HTTP API. Cover correctness, tests and security. Continue through all 150; do not summarize."
    configurations = [("off", False, 0, 512), ("small", True, 256, 768), ("medium", True, 512, 1024)]
    runs = []
    for label, thinking, budget, maximum in configurations:
        # Same prompt and model, a first and warm application-session run.
        # Neither this process nor iOS can establish a physically cold page cache.
        for repetition in range(2):
            start = time.monotonic()
            response = client.json("/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "temperature": 0, "seed": 42, "enable_thinking": thinking,
                "thinking_budget": budget, "max_tokens": maximum})
            wall = time.monotonic() - start
            usage = response.get("usage", {})
            metrics = usage.get("flash_local")
            tokens = usage.get("completion_tokens", 0)
            sample = {"configuration": label, "repetition": repetition, "http_wall_seconds": wall, "usage": usage,
                      "minimum_128_tokens_met": type(tokens) is int and tokens >= 128,
                      "os_cache_coldness_controlled": False, "same_model_comparison": True}
            if metrics is not None:
                passes, seconds = metrics.get("decode_forward_passes", 0), metrics.get("decode_seconds", 0)
                sample["decode_tokens_per_second"] = passes / seconds if seconds > 0 else None
                if passes:
                    sample["read_bytes_per_decode_pass"] = metrics.get("decode_io_bytes", 0) / passes
            runs.append(sample)
    return runs


def verify(client, model, include_benchmark=False):
    report = {"schema_version": 1, "model": model, "client": "Python stdlib / Kali-compatible",
              "physical_device_proven_by_client": False, "network_inference_location_independently_verified": False,
              "cross_model_benchmark_complete": False, "checks": {}}
    models = client.json("/models")
    if model not in [entry.get("id") for entry in models.get("data", [])]:
        raise ValueError("Requested model is not advertised by the live server")
    report["checks"]["models"] = True
    invalid = LocalClient(client.base, "known-invalid-test-key-" * 3, timeout=15)
    try:
        invalid.json("/models")
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise
        report["checks"]["authentication_rejects_wrong_key"] = True
        error.close()
    else:
        raise ValueError("Server accepted an invalid access key")
    report["tool_round_trips"] = [tool_round_trip(client, model), tool_round_trip(client, model, streaming=True)]
    report["checks"]["tool_calls_and_results"] = all(run["tool_result_receipt_observed"] for run in report["tool_round_trips"])
    if include_benchmark:
        report["benchmark_runs"] = benchmark(client, model)
    # Device identity is a server measurement, clearly distinguished from
    # independent physical verification (which requires the operator's phone).
    metrics = [value.get("followup_usage", {}).get("flash_local", {}) for value in report["tool_round_trips"]]
    report["server_reported_iphone18_2"] = all(value.get("platform") == "iOSDevice" and value.get("device_identifier") == "iPhone18,2" for value in metrics)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", required=True, help="New private report file, never overwritten")
    parser.add_argument("--benchmark", action="store_true", help="Six real runs; allow several minutes and keep the phone in the foreground")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        try:
            client = LocalClient(args.base_url, os.environ.get("FLASH_LOCAL_API_KEY", ""), timeout=1800)
            report = verify(client, args.model, args.benchmark)
        except (ValueError, OSError, KeyError, TypeError, urllib.error.URLError) as error:
            report = {"status": "failed", "error_type": type(error).__name__, "verification_complete": False}
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    success = bool(report.get("checks", {}).get("tool_calls_and_results"))
    print(json.dumps({"checks_passed": success, "report": str(output), "phone_testing_required": True}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
