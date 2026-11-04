#!/usr/bin/env python3
"""Replay the released Kinetix actor repair from pinned, read-only inputs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np


SOURCE_REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
SOURCE_FILES = {
    "runs/oneshot_repair/reply/prompt.txt": "2ed66051f4fa554d43d0163eb48e4256dc80aef78d875a218826beefeb7d0559",
    "runs/oneshot_repair/reply/answer.md": "dc295adb701bb943a6a324fd7fd4aeb97bb69a2d8fb73b9472e3a9371275f959",
    "networks/ppo_final_seed0.npz": "b0cdbccd5b4751d5a6dc4cb0deb18a036e6d08d71ad34802cea44dd26c6f4d11",
    "networks/edited_actor.npz": "610d62dbacdb16389f698cdf7d3bf39b185e410b8c6e03a1ac4833768066060a",
}
LAYERS = ("embed", *(f"actor{i}" for i in range(5)), "actor_head")
SHAPES = {
    f"{layer}.{part}": shape
    for layer, width_in, width_out in zip(LAYERS, (679, *([128] * 6)), (*([128] * 6), 16), strict=True)
    for part, shape in (("W", (width_in, width_out)), ("b", (width_out,)))
}
HEADER = re.compile(r"(\w+\.[Wb]) \(([0-9]+(?: x [0-9]+)?)\), scale (1e-[0-9]+):.*")
EDIT = re.compile(r"(\w+\.[Wb])\[([0-9:, ]+)\]\s*(=|\+=|-=|\*=)\s*(.*)")
NUMBER = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")


def read_pinned(path, expected):
    with path.open("rb") as stream:
        data = stream.read(2_000_000)
        if stream.read(1):
            raise ValueError(f"source exceeds size bound: {path.name}")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"source digest mismatch: {path.name}")
    return data


def parse_prompt_weights(prompt, shapes=SHAPES):
    _, marker, weights = prompt.partition("THE WEIGHTS (complete actor).")
    if not marker:
        raise ValueError("missing actor weights")
    lines = iter(line for line in weights.splitlines()[1:] if line.strip())
    result = {}
    for line in lines:
        match = HEADER.fullmatch(line)
        if not match:
            raise ValueError(f"invalid actor tensor header: {line[:60]}")
        name, dims, scale = match.groups()
        shape = tuple(map(int, dims.split(" x ")))
        if name in result or shapes.get(name) != shape:
            raise ValueError(f"unexpected or repeated actor tensor: {name}")
        rows = []
        for row in range(shape[0] if len(shape) == 2 else 1):
            values = next(lines, "")
            if len(shape) == 2:
                prefix, _, values = values.partition(":")
                if prefix != str(row):
                    raise ValueError(f"invalid row {row} of {name}")
            if not re.fullmatch(r"\s*-?\d+(?:\s+-?\d+)*\s*", values):
                raise ValueError(f"invalid weights in {name}")
            data = [int(value) for value in values.split()]
            if len(data) != shape[-1]:
                raise ValueError(f"invalid row width for {name}")
            rows.append(data)
        result[name] = np.asarray(rows, dtype="<f8").reshape(shape) * float(scale)
    if result.keys() != shapes.keys():
        raise ValueError("missing actor tensors")
    return result


def tensor_digest(actor):
    digest = hashlib.sha256()
    for name, array in sorted(actor.items()):
        digest.update(json.dumps((name, array.shape, array.dtype.str)).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def parse_selector(spec, size):
    if ":" not in spec:
        index = int(spec)
        if not 0 <= index < size:
            raise ValueError("edit index out of bounds")
        return index
    start, end = spec.split(":")
    lo, hi = int(start or 0), int(end or size)
    if not 0 <= lo < hi <= size:
        raise ValueError("edit range out of bounds or empty")
    return slice(lo, hi)


def edit_number(token):
    if not NUMBER.fullmatch(token):
        raise ValueError("invalid edit number")
    value = float(token)
    if abs(value) > 1_000_000:
        raise ValueError("edit number exceeds bound")
    return value


def apply_edits(actor, block):
    """Apply only the released numeric syntax; return a guarded reverse journal."""
    candidate = {name: tensor.copy() for name, tensor in actor.items()}
    journal = []
    for raw in block.splitlines():
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        match = EDIT.fullmatch(line)
        if not match:
            raise ValueError(f"unsupported edit: {line}")
        name, axes, op, literal = match.groups()
        if name not in candidate:
            raise ValueError(f"unknown edit tensor: {name}")
        target = candidate[name]
        key = tuple(parse_selector(axis.strip(), size)
                    for axis, size in zip(axes.split(","), target.shape, strict=True))
        before = target[key].copy()
        if literal.startswith("[") and literal.endswith("]"):
            value = np.array([edit_number(part.strip()) for part in literal[1:-1].split(",")])
            if value.shape != before.shape:
                raise ValueError("explicit edit values must exactly match selection shape")
        else:
            value = edit_number(literal)
        if op == "=":
            target[key] = value
        elif op == "+=":
            target[key] = before + value
        elif op == "-=":
            target[key] = before - value
        else:
            target[key] = before * value
        if not np.isfinite(target[key]).all():
            raise ValueError("edit produced non-finite weights")
        journal.append((name, key, before))
    if not journal:
        raise ValueError("no edits found")

    base_hash, candidate_hash = tensor_digest(actor), tensor_digest(candidate)

    def undo(edited):
        if tensor_digest(edited) != candidate_hash:
            raise ValueError("cannot roll back a different candidate")
        restored = {name: tensor.copy() for name, tensor in edited.items()}
        for name, key, previous in reversed(journal):
            restored[name][key] = previous
        if tensor_digest(restored) != base_hash:
            raise ValueError("restored actor differs from baseline")
        return restored

    return candidate, undo, len(journal)


def compare_full_checkpoint(baseline, prompt, checkpoint):
    scales = {match[1]: float(match[3]) for line in prompt.splitlines()
              if (match := HEADER.fullmatch(line))}
    baseline_with_signed_zeros = {}
    module = "params/GeneralActorCriticRNN_0"
    for name, rounded in baseline.items():
        layer, kind = name.split(".")
        prefix = ("params/Dense_0" if layer == "embed" else
                  f"{module}/Dense_0" if layer == "actor_head" else
                  f"{module}/DenseAndActivation_{layer[-1]}/Dense_0")
        full = checkpoint[f"{prefix}/{'kernel' if kind == 'W' else 'bias'}"]
        # The published prompt stores one scale per tensor, including for biases.
        quantized = np.rint(full.astype("<f8") / scales[name]) * scales[name]
        if full.dtype != np.float32 or not np.array_equal(quantized, rounded):
            raise ValueError(f"prompt weights differ from rounded PPO tensor: {name}")
        # Integer text erases the sign of rounded negative zeros; the PPO weights restore it.
        baseline_with_signed_zeros[name] = quantized
    return baseline_with_signed_zeros


def replay(root):
    data = {name: read_pinned(root / name, digest) for name, digest in SOURCE_FILES.items()}
    prompt = data["runs/oneshot_repair/reply/prompt.txt"]
    answer = data["runs/oneshot_repair/reply/answer.md"]
    checkpoint = data["networks/ppo_final_seed0.npz"]
    reference = data["networks/edited_actor.npz"]
    prompt = prompt.decode()
    base = parse_prompt_weights(prompt)
    with np.load(io.BytesIO(checkpoint), allow_pickle=False) as ppo:
        base = compare_full_checkpoint(base, prompt, ppo)
    blocks = re.findall(r"^\x60\x60\x60edits\n(.*?)^\x60\x60\x60\s*$",
                        answer.decode(), flags=re.MULTILINE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("expected one released edit block")
    edited, undo, count = apply_edits(base, blocks[0])
    with np.load(io.BytesIO(reference), allow_pickle=False) as published:
        if set(published.files) != set(edited) or any(
            edited[name].dtype != published[name].dtype or
            edited[name].shape != published[name].shape or
            edited[name].tobytes() != published[name].tobytes()
            for name in edited
        ):
            raise ValueError("replayed actor differs from the published edited actor")
    restored = undo(edited)
    total = sum(tensor.size for tensor in base.values())
    changed_by_tensor = {name: int(np.count_nonzero(base[name] != edited[name])) for name in base}
    changed = sum(changed_by_tensor.values())
    return base, edited, restored, {
        "source_revision": SOURCE_REVISION,
        "source_file_sha256": SOURCE_FILES,
        "method": "replay of published direct weight edits, with no gradients or training",
        "published_edited_actor_byte_match": True,
        "rounded_actor_matches_full_checkpoint": True,
        "rollback_byte_match": True,
        "edit_statements": count,
        "actor_entries": total,
        "changed_entries": changed,
        "changed_fraction": changed / total,
        "changed_by_tensor": changed_by_tensor,
        "actor_sha256": {
            "base": tensor_digest(base), "edited": tensor_digest(edited), "restored": tensor_digest(restored),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="released exp4_fix_car directory")
    parser.add_argument("--episodes", type=int, default=0,
                        help="paired CPU Kinetix rollouts; a multiple of eight up to 256 (default: 0)")
    parser.add_argument("--seed", type=int, default=20261104, help="paired action-sampling seed")
    args = parser.parse_args()
    if args.episodes not in range(0, 257, 8):
        parser.error("--episodes must be a multiple of eight in 0..256")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    try:
        base, edited, restored, report = replay(args.experiment)
        if args.episodes:
            from evaluate_kinetix_weight_edit import evaluate
            report["independent_kinetix_rollouts"] = evaluate(
                {"base": base, "edited": edited, "restored": restored}, args.episodes, args.seed
            )
    except (OSError, KeyError, ValueError, ImportError) as exc:
        parser.exit(1, f"replay failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
