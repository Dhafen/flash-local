"""Regenerable synthetic graph fixture, not checkpoint/model distribution.

This writer emits deterministic random FP32 weights. The wider --mixed graph
is subsequently encoded with canonical GGML quantizers by the C++ harness.
Includes HC, PLE, three GDN layers,
one dense attention layer, routed/shared experts and EOS hash history. The
separate pinned llama.cpp harness supplies expected logits, not this writer.
"""
import hashlib
import json
import math
import pathlib
import random
import struct
import argparse


def main(chat=False, mixed=False, ornith=False):
    if chat and mixed:
        raise ValueError("Mixed fixture has a separate graph-only vocabulary")
    hidden, width, rank, ple = (256, 256, 32, 128) if mixed else (16, 8, 4, 8)
    # This graph has two PLE heads; their concatenation feeds a hidden-wide
    # projection in canonical qwen4exp, rather than an arbitrary input width.
    if 2 * ple != hidden:
        raise ValueError("Concatenated PLE heads must match hidden width")
    wide = hidden * 2
    key_heads, value_heads = ((2, 4) if ornith else (2, 6)) if mixed else (1, 1)
    value_width = value_heads * 128
    channels = (2 * key_heads + value_heads) * 128
    rng = random.Random(20261015)
    architecture = "qwen35moe" if ornith else "qwen4exp"
    meta = {"general.architecture": (8, architecture), "general.name": (8, "synthetic-graph-test"),
            "tokenizer.ggml.model": (8, "none")}
    vocabulary_size = 262 if chat else 16
    if chat:
        from make_tokenizer_reference_fixture import metadata_prefix
        root = pathlib.Path(__file__).resolve().parents[1]
        header = (root / (".test-fixtures/ornith.header" if ornith else ".test-fixtures/ream.header")).read_bytes()
        if ornith:
            from fetch_ornith_metadata import HEADER_SHA256 as ORNITH_HEADER_SHA256
            _, template = metadata_prefix(header, expected_sha256=ORNITH_HEADER_SHA256)
        else:
            _, template = metadata_prefix(header)
        if not template:
            raise RuntimeError("Pinned chat template missing")
        direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
        extra, alphabet = 256, []
        for byte in range(256):
            alphabet.append(chr(byte if byte in direct else extra))
            if byte not in direct: extra += 1
        meta.update({
            "tokenizer.ggml.model": (8, "gpt2"), "tokenizer.ggml.pre": (8, "qwen35"),
            "tokenizer.ggml.tokens": (9, (8, alphabet + ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "ab"])),
            "tokenizer.ggml.token_type": (9, (5, [1] * 256 + [3, 3, 3, 4, 4, 1])),
            "tokenizer.ggml.merges": (9, (8, ["a b"])),
            "tokenizer.ggml.bos_token_id": (4, 258), "tokenizer.ggml.eos_token_id": (4, 257),
            "tokenizer.ggml.add_bos_token": (7, False), "tokenizer.ggml.add_eos_token": (7, False),
            "tokenizer.chat_template": (8, template),
        })
    integers = {
        "block_count": 4, "embedding_length": hidden, "vocab_size": vocabulary_size, "context_length": 4096,
        "feed_forward_length": width, "attention.head_count": 2, "attention.head_count_kv": 1,
        "attention.key_length": 8, "attention.value_length": 8, "rope.dimension_count": 2,
        "expert_count": 3, "expert_used_count": 2, "expert_feed_forward_length": width,
        "expert_shared_feed_forward_length": width, "ssm.conv_kernel": 2, "ssm.state_size": 128,
        "ssm.group_count": key_heads, "ssm.time_step_rank": value_heads, "ssm.inner_size": value_width,
        "full_attention_interval": 4, "hyper_connection.count": 2, "hyper_connection.low_rank": rank,
        "attention.indexer.head_count": 1, "attention.indexer.key_length": 8,
        "attention.indexer.top_k": 4, "ple.ngram_size": 3, "ple.heads_per_ngram": 1,
        "ple.conv_kernel": 2, "ple.eos_token_id": 15, "embedding_length_per_layer_input": ple,
    }
    if ornith:
        integers = {key: value for key, value in integers.items()
                    if not key.startswith(("hyper_connection.", "attention.indexer.", "ple."))
                    and key != "embedding_length_per_layer_input"}
        integers.update(block_count=5, nextn_predict_layers=1)
    for key, value in integers.items(): meta[architecture + "." + key] = (4, value)
    for key, value in {"rope.freq_base": 10000.0, "attention.layer_norm_rms_epsilon": 1e-6}.items():
        meta[architecture + "." + key] = (6, value)
    for key, values, kind in [
        ("rope.dimension_sections", [1, 0, 0, 0], 4),
        ("attention.compress_ratios", [0, 0, 0, 0], 4), ("ple.layers", [1], 4),
        ("ple.layer_multipliers", [3, 5, 7], 10),
        ("ple.head_offsets", [0, 5], 10), ("ple.head_vocab_sizes", [5, 5], 10),
    ]:
        if not ornith or key == "rope.dimension_sections":
            meta[architecture + "." + key] = (9, (kind, values))
    tensors = []

    def tensor(name, dimensions, constant=None):
        count = math.prod(dimensions)
        values = [constant if constant is not None else rng.uniform(-0.08, 0.08) * math.sqrt(16 / hidden) for _ in range(count)]
        tensors.append((name, dimensions, struct.pack("<" + "f" * count, *values)))

    def hc(prefix, injection=True):
        tensor(prefix + "norm.weight", [wide], 1.0)
        tensor(prefix + "down.weight", [wide, rank])
        tensor(prefix + "up.weight", [rank, wide])
        if injection: tensor(prefix + "inject.weight", [wide, 2])

    tensor("token_embd.weight", [hidden, vocabulary_size])
    tensor("output.weight", [hidden, vocabulary_size])
    if ornith:
        tensor("output_norm.weight", [hidden], 1.0)
    else:
        tensor("per_layer_token_embd.weight", [ple, 10])
        hc("output_hc_", False)
    for layer in range(5 if ornith else 4):
        p = f"blk.{layer}."
        if ornith:
            tensor(p + "attn_norm.weight", [hidden], 0.9)
            tensor(p + "post_attention_norm.weight", [hidden], 1.1)
        else:
            hc(p + "hc_attn_"); hc(p + "hc_ffn_")
        tensor(p + "ffn_gate_inp.weight", [hidden, 3])
        tensor(p + "ffn_gate_inp_shexp.weight", [hidden])
        for projection in ["gate", "up"]:
            tensor(p + f"ffn_{projection}_exps.weight", [hidden, width, 3])
            tensor(p + f"ffn_{projection}_shexp.weight", [hidden, width])
        tensor(p + "ffn_down_exps.weight", [width, hidden, 3])
        tensor(p + "ffn_down_shexp.weight", [width, hidden])
        if layer == 3 or (ornith and layer == 4):
            tensor(p + "attn_q.weight", [hidden, 32])
            tensor(p + "attn_k.weight", [hidden, 8]); tensor(p + "attn_v.weight", [hidden, 8])
            tensor(p + "attn_output.weight", [16, hidden])
            tensor(p + "attn_q_norm.weight", [8], 1.0); tensor(p + "attn_k_norm.weight", [8], 1.0)
            if not ornith:
                tensor(p + "indexer.q_proj.weight", [hidden, 8]); tensor(p + "indexer.k_proj.weight", [hidden, 8])
                tensor(p + "indexer.q_norm.weight", [8], 1.0); tensor(p + "indexer.k_norm.weight", [8], 1.0)
        else:
            tensor(p + "attn_qkv.weight", [hidden, channels]); tensor(p + "attn_gate.weight", [hidden, value_width])
            tensor(p + "ssm_alpha.weight", [hidden, value_heads]); tensor(p + "ssm_beta.weight", [hidden, value_heads])
            tensor(p + "ssm_conv1d.weight", [2, channels]); tensor(p + "ssm_out.weight", [value_width, hidden])
            tensor(p + "ssm_a", [value_heads], -0.25); tensor(p + "ssm_dt.bias", [value_heads], -1.0)
            tensor(p + "ssm_norm.weight", [128], 1.0)
        if layer == 1 and not ornith:
            tensor(p + "ple_key.weight", [2 * ple, wide]); tensor(p + "ple_value.weight", [2 * ple, hidden])
            for norm in ["key", "query", "conv"]: tensor(p + f"ple_norm_{norm}.weight", [wide], 1.0)
            tensor(p + "ple_conv1d.weight", [2, wide])
        if ornith and layer == 4:
            tensor(p + "nextn.eh_proj.weight", [2 * hidden, hidden])
            for norm in ["enorm", "hnorm", "shared_head_norm"]:
                tensor(p + f"nextn.{norm}.weight", [hidden], 1.0)

    def text(value):
        raw = value.encode(); return struct.pack("<Q", len(raw)) + raw

    def value(kind, item):
        if kind == 8: return text(item)
        if kind == 9:
            element, entries = item
            return struct.pack("<IQ", element, len(entries)) + b"".join(value(element, e) for e in entries)
        return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q"}[kind], item)

    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(meta)))
    for key, (kind, item) in meta.items(): header += text(key) + struct.pack("<I", kind) + value(kind, item)
    payload = bytearray()
    for name, dimensions, raw in tensors:
        payload += bytes((-len(payload)) % 32)
        header += text(name) + struct.pack("<I", len(dimensions))
        header += b"".join(struct.pack("<Q", d) for d in dimensions)
        header += struct.pack("<IQ", 0, len(payload))
        payload += raw
    header += bytes((-len(header)) % 32)
    model = bytes(header + payload)
    folder = pathlib.Path(__file__).resolve().parents[1] / ".test-fixtures"
    folder.mkdir(exist_ok=True)
    family = "ornith" if ornith else "qwen4exp"
    stem = f"synthetic-{family}-mixed-source" if mixed else (f"synthetic-{family}-chat" if chat else f"synthetic-{family}")
    (folder / (stem + ".gguf")).write_bytes(model)
    manifest = dict(fileName=stem + ".gguf", fileByteCount=len(model),
                    sha256=hashlib.sha256(model).hexdigest(), tokens=[1, 2, 3, 15, 4], contextTokens=128 if chat else 32)
    if chat:
        manifest.update(messages=[{"role": "user", "content": "Hi"}], maximumNewTokens=8)
    (folder / (stem + ".json")).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(tensors)} random synthetic tensors, {len(model)} bytes; no trained weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="Add tiny byte vocabulary and pinned chat template")
    parser.add_argument("--mixed", action="store_true", help="Generate wider random graph for canonical mixed quantization")
    parser.add_argument("--ornith", action="store_true", help="Qwen3.5 MoE trunk with an unused MTP block; no HC/PLE")
    args = parser.parse_args()
    main(chat=args.chat, mixed=args.mixed, ornith=args.ornith)
