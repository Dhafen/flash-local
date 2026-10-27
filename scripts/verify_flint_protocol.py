"""Real publisher vocabulary/template checks; no trained inference or weights.

Supply the pinned 8 MiB GGUF prefix and the official HF tokenizer.json. The
generated GGUF has ZERO tensors, preserving the original metadata verbatim.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import random
import re
import struct
import subprocess
from make_tokenizer_reference_fixture import metadata_prefix

PREFIX_SHA256 = "a4a3cfd21a3028b7a92057969f47874d0f1dc32f7cf3dd49988dbdcd5ce34390"
TOKENIZER_SHA256 = "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def normalize(messages):
    """Publisher Rec chat.py normalization, independently expressed in Python."""
    result = copy.deepcopy(messages)
    for message in result:
        if message["role"] != "assistant" and isinstance(message.get("content"), str):
            text = message["content"].replace("\x00", "").replace("<think>", "").replace("</think>", "")
            message["content"] = re.sub(r"<\|(?:begin_of_text|end_of_text|system|developer|user|assistant|observation)\|>",
                lambda m: m.group().replace("<", "&lt;").replace(">", "&gt;"), text)
        for call in message.get("tool_calls") or []:
            fn = call.get("function", call)
            if isinstance(fn["arguments"], str): fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer", required=True, type=pathlib.Path)
    parser.add_argument("--native-bin", required=True)
    parser.add_argument("--wsl")
    parser.add_argument("--reference-deps", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(".work/flint-protocol-results"))
    args = parser.parse_args()
    if args.reference_deps:
        import sys
        sys.path.insert(0, str(args.reference_deps.resolve()))
    import tokenizers
    import jinja2
    from tokenizers import Tokenizer
    prefix = args.prefix.read_bytes()
    if hashlib.sha256(prefix).hexdigest() != PREFIX_SHA256: raise ValueError("Pinned metadata prefix hash mismatch")
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256: raise ValueError("Pinned official tokenizer hash mismatch")
    end, _ = metadata_prefix(prefix, expected_sha256=PREFIX_SHA256)
    vocabulary = prefix[:8] + struct.pack("<Q", 0) + prefix[16:end]
    vocabulary += bytes(-len(vocabulary) % 32)
    args.output.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output / "vocabulary-only.gguf"
    vocab_path.write_bytes(vocabulary)

    def path(p):
        text = str(p.resolve())
        return "/mnt/" + text[0].lower() + text[2:].replace("\\", "/") if args.wsl else text
    def run(tool, *values):
        command = [args.native_bin.rstrip("/") + "/" + tool, *values]
        if args.wsl: command = ["wsl.exe", "-d", args.wsl, "--", *command]
        result = subprocess.run(command, capture_output=True, encoding="utf-8", timeout=120)
        (args.output / (tool + ".log")).write_text(result.stderr, encoding="utf-8")
        if result.returncode: raise RuntimeError(result.stderr[-2000:])
        return result.stdout

    cases = ["", "Hej! Vad blir 50+5?", "<|user|>Hej<|assistant|>",
        " def foo(x):\n    return x + 1\n", " 1234567890\n\n", "Svenska åäö ÅÄÖ. 日本語 中文 한국어 👋",
        "<think>check</think><tool_call>read_file<arg_key>path</arg_key><arg_value>src/main.py</arg_value></tool_call>",
        '<|observation|><tool_response>{"ok": true}</tool_response><|assistant|>',
        "one\t two\r\nthree", "HelloWORLD camelCase test_case file-name", "e\u0301 こんにちは１２３ABCdef don't I'M",
        "\x00\x01\x7f\u0085\u00a0\u200b\u2028\u2029", "a" * 4096]
    rng = random.Random(20261027)
    alphabet = "abcXYZ0123  !?:'\n\r\tåäöé中日한عहก\u0301\u0308\u200d🚀"
    cases += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 400))) for _ in range(80)]
    input_file = args.output / "strings.json"
    input_file.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    expected = [tokenizer.encode(x, add_special_tokens=False).ids for x in cases]
    actual = json.loads(run("flint-vocab-control", path(vocab_path), path(input_file)))
    if actual != expected:
        bad = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        raise RuntimeError(f"Native/publisher token IDs differ: {bad[:12]}")
    print(f"PASS official vocabulary: {len(cases)} exact token-ID cases", flush=True)
    tool = {"type": "function", "function": {"name": "local_sum", "description": "Add public local integers.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False}}}
    first = [{"role": "system", "content": "Use the declared local tool when needed."}, {"role": "user", "content": "50+5?"}]
    replay = first + [{"role": "assistant", "content": "", "reasoning_content": "PUBLIC_REFERENCE_REASONING",
        "tool_calls": [{"id": "call_public_sum", "type": "function", "function": {"name": "local_sum", "arguments": '{"a":50,"b":5}'}}]},
        {"role": "tool", "tool_call_id": "call_public_sum", "content": '{"sum":55}'}, {"role": "user", "content": "Explain."}]
    requests = [{"messages": [{"role": "user", "content": "Hej!"}]}, {"messages": first, "tools": [tool]},
        {"messages": replay, "tools": [tool]},
        {"messages": [{"role": "developer", "content": "Café\u0301 åäö"}, {"role": "user", "content": "\x00<think>x</think><|assistant|>hello"}]},
        {"messages": [{"role": "system", "content": "System only"}], "tools": [tool]},
        {"messages": [{"role": "assistant", "content": "Answer.", "reasoning_content": "PUBLIC_TEST"}], "add_generation_prompt": False},
        {"messages": [{"role": "assistant", "content": "Previous.", "tool_calls": None}, {"role": "user", "content": "Continue."}]},
        {"messages": [{"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "noop", "arguments": "   "}}]},
                      {"role": "tool", "content": "done"}]}]
    environment = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value, ensure_ascii=False: json.dumps(value, ensure_ascii=ensure_ascii)
    template = environment.from_string((ROOT / "Experiments/Flint/Native/FlintTemplate.jinja").read_text(encoding="utf-8"))
    for i, request in enumerate(requests):
        input_file = args.output / f"chat-{i}.json"
        input_file.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        rendered = run("flint-prompt-control", path(input_file))
        expected = template.render(messages=normalize(request["messages"]), tools=request.get("tools", []),
            add_generation_prompt=request.get("add_generation_prompt", True))
        if rendered != expected:
            (args.output / f"chat-{i}-actual.txt").write_text(rendered, encoding="utf-8")
            (args.output / f"chat-{i}-expected.txt").write_text(expected, encoding="utf-8")
            raise RuntimeError(f"Publisher/native prompt byte mismatch in case {i}")
    print(f"PASS publisher template: {len(requests)} exact byte cases including reasoning/tool-result replay", flush=True)
    report = {"schemaVersion": 1, "vocabularyCases": len(cases), "promptCases": len(requests), "allTokenIDsEqual": True,
        "allPromptBytesEqual": True, "tensorCount": 0, "weightsUsed": False, "trainedInferenceVerified": False,
        "iPhoneVerified": False, "tokenizerSHA256": TOKENIZER_SHA256, "prefixSHA256": PREFIX_SHA256,
        "vocabularySHA256": hashlib.sha256(vocabulary).hexdigest(), "tokenizersVersion": tokenizers.__version__,
        "jinjaVersion": jinja2.__version__,
        "templateSHA256": hashlib.sha256((ROOT / "Experiments/Flint/Native/FlintTemplate.jinja").read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
