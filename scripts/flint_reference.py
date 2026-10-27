"""Independent NumPy/FP64 graph control for SMALL synthetic Flint GGUFs.

Implements the published RecursiveSI/rec model equations, not llama's graph.
No trained model loader: a 64 MiB input cap prevents a full-model RAM fallback.
"""
import math
import pathlib
import struct

import numpy as np


def read_small_gguf(path):
    path = pathlib.Path(path)
    if path.stat().st_size > 64 << 20: raise ValueError("Synthetic controls only; 64 MiB file cap")
    raw = path.read_bytes()
    position = 0

    def unpack(fmt):
        nonlocal position
        size = struct.calcsize(fmt)
        out = struct.unpack_from(fmt, raw, position)
        position += size
        return out[0] if len(out) == 1 else out

    def string():
        nonlocal position
        size = unpack("<Q")
        text = raw[position:position + size].decode("utf-8")
        position += size
        return text

    def value(kind):
        if kind == 8: return string()
        if kind == 9:
            element, count = unpack("<IQ")
            return [value(element) for _ in range(count)]
        return unpack({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[kind])

    magic, version, count, keys = unpack("<4sIQQ")
    if magic != b"GGUF" or version != 3: raise ValueError("Expected GGUF v3")
    metadata = {}
    for _ in range(keys):
        name = string()
        metadata[name] = value(unpack("<I"))
    descriptors = []
    for _ in range(count):
        name = string()
        dims = [unpack("<Q") for _ in range(unpack("<I"))]
        kind, offset = unpack("<IQ")
        descriptors.append((name, dims, kind, offset))
    alignment = metadata.get("general.alignment", 32)
    data_start = position + (-position % alignment)
    weights = {}
    for name, dims, kind, offset in descriptors:
        n = math.prod(dims)
        start = data_start + offset
        if kind == 0:
            data = np.frombuffer(raw, dtype="<f4", count=n, offset=start).copy()
        elif kind == 3:
            if n % 32: raise ValueError("Unaligned Q4_1 tensor")
            blocks = np.frombuffer(raw, dtype=np.uint8, count=n // 32 * 20, offset=start).reshape(-1, 20)
            d = blocks[:, :2].copy().view("<f2").astype(np.float32)
            m = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
            q = np.concatenate([blocks[:, 4:] & 15, blocks[:, 4:] >> 4], axis=1).astype(np.float32)
            data = (d * q + m).reshape(-1)
        else: raise ValueError(f"Unsupported synthetic quantization {kind}")
        weights[name] = data.reshape(tuple(reversed(dims))).astype(np.float64)
    return metadata, weights


def evaluate(path, tokens):
    meta, w = read_small_gguf(path)
    cfg = lambda key: meta["flint." + key]
    layers, head, nq, nk = cfg("block_count"), cfg("attention.key_length"), cfg("attention.head_count"), cfg("attention.head_count_kv")
    keys, values, logits, routes = [[] for _ in range(layers)], [[] for _ in range(layers)], [], []
    epsilon = cfg("attention.layer_norm_rms_epsilon")

    def norm(x, weight): return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * weight
    def silu(x): return x / (1.0 + np.exp(np.clip(-x, -700, 700)))
    def rope(x, position):
        freq = position * cfg("rope.freq_base") ** (-np.arange(head // 2) * 2.0 / head)
        a, b = np.split(x, 2, axis=-1)
        return np.concatenate([a * np.cos(freq) - b * np.sin(freq), a * np.sin(freq) + b * np.cos(freq)], axis=-1)

    for position, token in enumerate(tokens):
        x = w["token_embd.weight"][token].copy()
        for il in range(layers):
            p = f"blk.{il}."
            h = norm(x, w[p + "attn_norm.weight"])
            q = norm((w[p + "attn_q.weight"] @ h).reshape(nq, head), w[p + "attn_q_norm.weight"])
            k = norm((w[p + "attn_k.weight"] @ h).reshape(nk, head), w[p + "attn_k_norm.weight"])
            v = (w[p + "attn_v.weight"] @ h).reshape(nk, head)
            if cfg("attention.rope_pattern")[il]: q, k = rope(q, position), rope(k, position)
            keys[il].append(k); values[il].append(v)
            first = max(0, position + 1 - cfg("attention.sliding_window")) if cfg("attention.sliding_window_pattern")[il] else 0
            ka, va = np.array(keys[il][first:]), np.array(values[il][first:])
            out = []
            for i in range(nq):
                kv = i // (nq // nk)
                scores = (ka[:, kv] @ q[i]) / math.sqrt(head)
                sink = w[p + "attn_sinks.weight"][i]
                maximum = max(sink, scores.max())
                probs = np.exp(scores - maximum)
                probs /= probs.sum() + math.exp(sink - maximum)
                out.append(probs @ va[:, kv])
            x += w[p + "attn_output.weight"] @ np.concatenate(out)
            h = norm(x, w[p + "ffn_norm.weight"])
            if il < cfg("leading_dense_block_count"):
                y = w[p + "ffn_down.weight"] @ (silu(w[p + "ffn_gate.weight"] @ h) * (w[p + "ffn_up.weight"] @ h))
            else:
                scores = 1 / (1 + np.exp(-(w[p + "ffn_gate_inp.weight"] @ h)))
                choice = scores + w.get(p + "exp_probs_b.bias", 0)
                selected = np.argsort(-choice, kind="stable")[:cfg("expert_used_count")]
                routes.append({"position": position, "layer": il, "experts": selected.tolist()})
                mixing = scores[selected] / scores[selected].sum() * cfg("expert_weights_scale")
                y = np.zeros_like(h)
                for expert, weight in zip(selected, mixing):
                    z = silu(w[p + "ffn_gate_exps.weight"][expert] @ h) * (w[p + "ffn_up_exps.weight"][expert] @ h)
                    y += weight * (w[p + "ffn_down_exps.weight"][expert] @ z)
                y += w[p + "ffn_down_shexp.weight"] @ (silu(w[p + "ffn_gate_shexp.weight"] @ h) * (w[p + "ffn_up_shexp.weight"] @ h))
            x += y
        logits.append(w["output.weight"] @ norm(x, w["output_norm.weight"]))
    return np.array(logits), routes
