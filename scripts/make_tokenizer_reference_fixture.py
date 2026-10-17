"""Create a vocabulary-only GGUF and public test strings from the pinned header.

No tensor payload, trained inference, user prompt or model download is involved.
The original metadata bytes (including vocabulary/types/merges) are unchanged.
"""
import hashlib
import json
import pathlib
import random
import struct

from fetch_ream_metadata import HEADER_SHA256


def metadata_prefix(header, expected_sha256=HEADER_SHA256):
    """Return exact KV boundary and chat template without retaining vocab arrays."""
    if hashlib.sha256(header).hexdigest() != expected_sha256:
        raise RuntimeError("Tokenizer source header hash mismatch")
    position = 24

    def integer(fmt):
        nonlocal position
        value = struct.unpack_from("<" + fmt, header, position)[0]
        position += struct.calcsize("<" + fmt)
        return value

    def string(retain=False):
        nonlocal position
        count = integer("Q")
        if count > 2 << 20 or position + count > len(header):
            raise RuntimeError("Invalid bounded string")
        start = position
        position += count
        return header[start:position].decode("utf-8") if retain else None

    def skip(kind):
        if kind == 8:
            string()
        elif kind == 9:
            element, count = integer("I"), integer("Q")
            if element == 9 or count > 1_000_000:
                raise RuntimeError("Invalid bounded array")
            for _ in range(count):
                skip(element)
        else:
            integer({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                     6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}[kind])

    metadata_count = struct.unpack_from("<Q", header, 16)[0]
    template = None
    for _ in range(metadata_count):
        key, kind = string(True), integer("I")
        if key == "tokenizer.chat_template":
            if kind != 8:
                raise RuntimeError("Chat template is not a string")
            template = string(True)
        else:
            skip(kind)
    return position, template


def main(ornith=False):
    root = pathlib.Path(__file__).resolve().parents[1]
    source = root / ".test-fixtures"
    if ornith:
        from fetch_ornith_metadata import HEADER_SHA256 as expected_hash
    else:
        expected_hash = HEADER_SHA256
    folder = source / "ornith-tokenizer" if ornith else source
    folder.mkdir(exist_ok=True)
    header = (source / ("ornith.header" if ornith else "ream.header")).read_bytes()
    position, _ = metadata_prefix(header, expected_sha256=expected_hash)
    # A legal metadata-only GGUF: all original KV bytes, zero tensors.
    vocabulary = header[:8] + struct.pack("<Q", 0) + header[16:position]
    vocabulary += bytes((-len(vocabulary)) % 32)
    (folder / "qwen35-vocabulary-only.gguf").write_bytes(vocabulary)

    cases = ["", "Hej världen!", "Hello world!", " don't DON'T I'm WE'LL you've",
             "1234567890 ٣٤५６７", "café cafe\u0301 Å A\u030a ö a\u0308",
             "中文测试。日本語のテスト 한국어 테스트", "Привет мир Ελληνικά",
             "مرحبا بالعالم नमस्ते दुनिया ภาษาไทย", "\t  \r\n\n  trailing   ",
             "👨‍👩‍👧‍👦 🚀 🇸🇪 🧑🏽‍💻", "\x00\x01\x7f\u0085\u00a0\u200b\u2028\u2029",
             "func fib(_ n: Int) -> Int { n < 2 ? n : fib(n-1) + fib(n-2) }\n",
             'SELECT * FROM users WHERE id = 42;\n{"key": [true, null, 1.2e-3]}',
             "https://example.org/a_b?q=x%20y&n=42#fragment", "a" * 4096,
             "abababab" * 512, " \n\t" * 100, "<example>ordinary text</example>",
             "\U00010000\U0001d400\U00020000\U000e0100"]
    rng = random.Random(20261015)
    alphabet = ("abcXYZ0123 '!?\n\r\t" + "åé中日한عहก" + "\u0301\u0308\u200d" + "🚀")
    cases += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 400))) for _ in range(80)]
    # Cover non-ASCII Unicode classes across planes, including combining marks.
    for start in [0x80, 0x300, 0x600, 0x900, 0xe00, 0x2000, 0x1d400, 0x1f300, 0x20000]:
        cases.append("".join(chr(start + rng.randrange(128)) for _ in range(96)))
    chats = [
        [{"role": "user", "content": "Hej!"}],
        [{"role": "system", "content": "  Answer briefly.\r\n"},
         {"role": "user", "content": "  café cafe\u0301 中文 🚀  "}],
        [{"role": "system", "content": " \t\n"},
         {"role": "user", "content": "One"}, {"role": "assistant", "content": " Two "},
         {"role": "user", "content": "Three"}],
        [{"role": "user", "content": "\u00a0边界\u00a0"}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": "function(a, b) { return a + b; }\n"},
         {"role": "assistant", "content": "A sum."},
         {"role": "user", "content": "What about subtraction?"}],
    ]
    for text in cases[20:36]:
        chats.append([{"role": "user", "content": text}])
    manifest = {"headerSHA256": expected_hash,
                "vocabularySHA256": hashlib.sha256(vocabulary).hexdigest(),
                "vocabularyByteCount": len(vocabulary), "cases": cases,
                "chats": [{"messages": messages, "enable_thinking": False,
                           "preserve_thinking": False, "add_generation_prompt": True} for messages in chats]}
    manifest["reasoningChats"] = [
        {"messages": [{"role": "user", "content": "First"},
                      {"role": "assistant", "content": "55",
                       "reasoning_content": "PUBLIC_REASONING_FIXTURE"},
                      {"role": "user", "content": "Continue"}],
         "enable_thinking": enabled, "preserve_thinking": True,
         "add_generation_prompt": True} for enabled in (False, True)]
    (folder / "qwen35-tokenizer-corpus.json").write_text(
        json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    binary = bytearray(struct.pack("<I", len(cases)))
    for text in cases:
        data = text.encode("utf-8")
        binary += struct.pack("<I", len(data)) + data
    (folder / "qwen35-tokenizer-corpus.bin").write_bytes(binary)
    print(f"Created {len(cases)} public tokenizer cases and {len(vocabulary)} metadata bytes; zero tensors")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ornith", action="store_true")
    main(ornith=parser.parse_args().ornith)
