"""Small deterministic UNTRAINED Flint graph, never a quality/performance claim."""
import argparse
import hashlib
import json
import math
import pathlib
import random
import struct


def string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def value(kind, item):
    if kind == 8: return string(item)
    if kind == 9:
        element, items = item
        return struct.pack("<IQ", element, len(items)) + b"".join(value(element, x) for x in items)
    return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q"}[kind], item)


def make(output, amplification=1.0, expert_count=16):
    output.mkdir(parents=True, exist_ok=True)
    if expert_count not in (16, 256): raise ValueError("Only bounded synthetic expert counts 16/256 are supported")
    hidden, head, qheads, kvheads, width, experts, layers, vocab = 64, 32, 4, 1, 64, expert_count, 6, 32
    swa = [0, 1, 1, 0, 1, 0]
    meta = {"general.architecture": (8, "flint"), "general.name": (8, "UNTRAINED Flint synthetic control"),
            "tokenizer.ggml.model": (8, "none")}
    for key, item in {"block_count": layers, "embedding_length": hidden, "vocab_size": vocab,
        "context_length": 512, "feed_forward_length": 128, "attention.head_count": qheads,
        "attention.head_count_kv": kvheads, "attention.key_length": head, "attention.value_length": head,
        "rope.dimension_count": head, "expert_count": experts, "expert_used_count": 6,
        "leading_dense_block_count": 2, "expert_feed_forward_length": width,
        "expert_shared_count": 2, "expert_shared_feed_forward_length": 128,
        "expert_gating_func": 2, "attention.sliding_window": 9}.items(): meta["flint." + key] = (4, item)
    for key, item in {"rope.freq_base": 10000.0, "attention.layer_norm_rms_epsilon": 1e-6,
                      "expert_weights_scale": 1.0}.items(): meta["flint." + key] = (6, item)
    meta["flint.expert_weights_norm"] = (7, True)
    for key in ["attention.sliding_window_pattern", "attention.rope_pattern"]:
        meta["flint." + key] = (9, (4, swa))
    rng = random.Random(20261027)
    tensors = []

    def tensor(name, dims, constant=None, scale=1.0):
        values = [constant if constant is not None else rng.uniform(-0.13, 0.13) * scale for _ in range(math.prod(dims))]
        tensors.append((name, dims, struct.pack("<" + "f" * len(values), *values)))

    tensor("token_embd.weight", [hidden, vocab])
    tensor("output.weight", [hidden, vocab])
    tensor("output_norm.weight", [hidden], 1.0)
    for il in range(layers):
        p = f"blk.{il}."
        tensor(p + "attn_norm.weight", [hidden], 1.0)
        tensor(p + "ffn_norm.weight", [hidden], 1.0)
        for family, size in [("q", qheads * head), ("k", kvheads * head), ("v", kvheads * head)]:
            tensor(p + f"attn_{family}.weight", [hidden, size])
        tensor(p + "attn_output.weight", [qheads * head, hidden])
        tensor(p + "attn_q_norm.weight", [head], 0.9)
        tensor(p + "attn_k_norm.weight", [head], 1.1)
        tensor(p + "attn_sinks.weight", [qheads], -0.7)
        if il < 2:
            for family in ["gate", "up"]: tensor(p + f"ffn_{family}.weight", [hidden, 128])
            tensor(p + "ffn_down.weight", [128, hidden])
        else:
            tensor(p + "ffn_gate_inp.weight", [hidden, experts])
            for family in ["gate", "up"]:
                tensor(p + f"ffn_{family}_exps.weight", [hidden, width, experts], scale=amplification)
                tensor(p + f"ffn_{family}_shexp.weight", [hidden, 128])
            tensor(p + "ffn_down_exps.weight", [width, hidden, experts], scale=1.0 / amplification**2)
            tensor(p + "ffn_down_shexp.weight", [128, hidden])
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(meta)))
    for key, (kind, item) in meta.items(): header += string(key) + struct.pack("<I", kind) + value(kind, item)
    payload, entries = bytearray(), []
    for name, dims, raw in tensors:
        payload += bytes(-len(payload) % 32)
        offset = len(payload)
        header += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        header += struct.pack("<IQ", 0, offset)
        payload += raw
        if "_exps.weight" in name:
            entries.append({"layer": int(name.split(".")[1]), "tensor_family": name.split(".")[2],
                "tensor_name": name, "quant_type": "f32", "exact_byte_length": len(raw),
                "bytes_per_expert": len(raw) // experts, "expert_stride": len(raw) // experts,
                "expert_count": experts, "repacked_offset": offset, "repacked_file": "synthetic-flint-f32.gguf"})
    header += bytes(-len(header) % 32)
    for entry in entries: entry["repacked_offset"] += len(header)
    model = bytes(header + payload)
    path = output / "synthetic-flint-f32.gguf"
    path.write_bytes(model)
    manifest = {"schema_version": 1, "experts_kind": "omni_moe_gguf", "layout": "gguf_tensor_major",
                "model": {"expert_count": experts, "expert_used_count": 6}, "entries": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"synthetic": True, "sha256": hashlib.sha256(model).hexdigest(), "fileBytes": len(model),
            "layers": layers, "experts": experts, "topK": 6, "slidingWindow": 9, "amplification": amplification}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(".work/flint-controls"))
    parser.add_argument("--amplification", type=float, default=1.0)
    parser.add_argument("--wide-experts", action="store_true", help="Use the actual 256-expert/top-6 router geometry")
    args = parser.parse_args()
    print(json.dumps(make(args.output, args.amplification, 256 if args.wide_experts else 16)))
