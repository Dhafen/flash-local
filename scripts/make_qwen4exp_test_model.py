"""Regenerable synthetic graph fixture, not checkpoint/model distribution.

All weights are deterministic random FP32. Includes HC, PLE, three GDN layers,
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


def main(chat=False):
    rng = random.Random(20261015)
    meta = {"general.architecture": (8, "qwen4exp"), "general.name": (8, "synthetic-graph-test"),
            "tokenizer.ggml.model": (8, "none")}
    vocabulary_size = 262 if chat else 16
    if chat:
        from make_tokenizer_reference_fixture import metadata_prefix
        root = pathlib.Path(__file__).resolve().parents[1]
        _, template = metadata_prefix((root / ".test-fixtures/ream.header").read_bytes())
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
            "tokenizer.ggml.token_type": (9, (4, [1] * 256 + [3, 3, 3, 4, 4, 1])),
            "tokenizer.ggml.merges": (9, (8, ["a b"])),
            "tokenizer.ggml.bos_token_id": (4, 258), "tokenizer.ggml.eos_token_id": (4, 257),
            "tokenizer.ggml.add_bos_token": (7, False), "tokenizer.ggml.add_eos_token": (7, False),
            "tokenizer.chat_template": (8, template),
        })
    integers = {
        "block_count": 4, "embedding_length": 16, "vocab_size": vocabulary_size, "context_length": 4096,
        "feed_forward_length": 8, "attention.head_count": 2, "attention.head_count_kv": 1,
        "attention.key_length": 8, "attention.value_length": 8, "rope.dimension_count": 2,
        "expert_count": 3, "expert_used_count": 2, "expert_feed_forward_length": 8,
        "expert_shared_feed_forward_length": 8, "ssm.conv_kernel": 2, "ssm.state_size": 128,
        "ssm.group_count": 1, "ssm.time_step_rank": 1, "ssm.inner_size": 128,
        "full_attention_interval": 4, "hyper_connection.count": 2, "hyper_connection.low_rank": 4,
        "attention.indexer.head_count": 1, "attention.indexer.key_length": 8,
        "attention.indexer.top_k": 4, "ple.ngram_size": 3, "ple.heads_per_ngram": 1,
        "ple.conv_kernel": 2, "ple.eos_token_id": 15, "embedding_length_per_layer_input": 8,
    }
    for key, value in integers.items(): meta["qwen4exp." + key] = (4, value)
    for key, value in {"rope.freq_base": 10000.0, "attention.layer_norm_rms_epsilon": 1e-6}.items():
        meta["qwen4exp." + key] = (6, value)
    for key, values, kind in [
        ("rope.dimension_sections", [1, 0, 0, 0], 4),
        ("attention.compress_ratios", [0, 0, 0, 0], 4), ("ple.layers", [1], 4),
        ("ple.layer_multipliers", [3, 5, 7], 10),
        ("ple.head_offsets", [0, 5], 10), ("ple.head_vocab_sizes", [5, 5], 10),
    ]: meta["qwen4exp." + key] = (9, (kind, values))
    tensors = []

    def tensor(name, dimensions, constant=None):
        count = math.prod(dimensions)
        values = [constant if constant is not None else rng.uniform(-0.08, 0.08) for _ in range(count)]
        tensors.append((name, dimensions, struct.pack("<" + "f" * count, *values)))

    def hc(prefix, injection=True):
        tensor(prefix + "norm.weight", [32], 1.0)
        tensor(prefix + "down.weight", [32, 4])
        tensor(prefix + "up.weight", [4, 32])
        if injection: tensor(prefix + "inject.weight", [32, 2])

    tensor("token_embd.weight", [16, vocabulary_size])
    tensor("output.weight", [16, vocabulary_size])
    tensor("per_layer_token_embd.weight", [8, 10])
    hc("output_hc_", False)
    for layer in range(4):
        p = f"blk.{layer}."
        hc(p + "hc_attn_"); hc(p + "hc_ffn_")
        tensor(p + "ffn_gate_inp.weight", [16, 3])
        tensor(p + "ffn_gate_inp_shexp.weight", [16])
        for projection in ["gate", "up"]:
            tensor(p + f"ffn_{projection}_exps.weight", [16, 8, 3])
            tensor(p + f"ffn_{projection}_shexp.weight", [16, 8])
        tensor(p + "ffn_down_exps.weight", [8, 16, 3])
        tensor(p + "ffn_down_shexp.weight", [8, 16])
        if layer == 3:
            tensor(p + "attn_q.weight", [16, 32])
            tensor(p + "attn_k.weight", [16, 8]); tensor(p + "attn_v.weight", [16, 8])
            tensor(p + "attn_output.weight", [16, 16])
            tensor(p + "attn_q_norm.weight", [8], 1.0); tensor(p + "attn_k_norm.weight", [8], 1.0)
            tensor(p + "indexer.q_proj.weight", [16, 8]); tensor(p + "indexer.k_proj.weight", [16, 8])
            tensor(p + "indexer.q_norm.weight", [8], 1.0); tensor(p + "indexer.k_norm.weight", [8], 1.0)
        else:
            tensor(p + "attn_qkv.weight", [16, 384]); tensor(p + "attn_gate.weight", [16, 128])
            tensor(p + "ssm_alpha.weight", [16, 1]); tensor(p + "ssm_beta.weight", [16, 1])
            tensor(p + "ssm_conv1d.weight", [2, 384]); tensor(p + "ssm_out.weight", [128, 16])
            tensor(p + "ssm_a", [1], -0.25); tensor(p + "ssm_dt.bias", [1], -1.0)
            tensor(p + "ssm_norm.weight", [128], 1.0)
        if layer == 1:
            tensor(p + "ple_key.weight", [16, 32]); tensor(p + "ple_value.weight", [16, 16])
            for norm in ["key", "query", "conv"]: tensor(p + f"ple_norm_{norm}.weight", [32], 1.0)
            tensor(p + "ple_conv1d.weight", [2, 32])

    def text(value):
        raw = value.encode(); return struct.pack("<Q", len(raw)) + raw

    def value(kind, item):
        if kind == 8: return text(item)
        if kind == 9:
            element, entries = item
            return struct.pack("<IQ", element, len(entries)) + b"".join(value(element, e) for e in entries)
        return struct.pack({4: "<I", 6: "<f", 7: "<?", 10: "<Q"}[kind], item)

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
    stem = "synthetic-qwen4exp-chat" if chat else "synthetic-qwen4exp"
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
    main(chat=parser.parse_args().chat)
