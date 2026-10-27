"""Real pinned Q4_1 slices vs independent NumPy, without full-model inference.

Rechecks the full SHA in 4 MiB buffers. Reads just 16 rows from each selected
expert/family, invokes canonical GGML CPU dequant/dot, then discards raw slices.
No tokenizer or model graph is loaded. These are format/operator controls, not
trained answer quality, iPhone performance, or end-to-end logit verification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from download_flint_model import identity, hash_prefix


def layout(path):
    with path.open("rb") as file: raw = file.read(8 << 20)
    at = 0
    def take(size):
        nonlocal at
        if size < 0 or at + size > len(raw): raise ValueError("GGUF metadata exceeds bounded header")
        result = raw[at:at+size]; at += size
        return result
    def unpack(fmt):
        result = struct.unpack(fmt, take(struct.calcsize(fmt)))
        return result[0] if len(result) == 1 else result
    def string():
        size = unpack("<Q")
        if size > 1 << 20: raise ValueError("Oversized GGUF string")
        return take(size).decode("utf-8")
    def value(kind, retain):
        if kind == 8: return string()
        if kind == 9:
            element, count = unpack("<IQ")
            if element == 9 or count > 300000: raise ValueError("Unsupported metadata array")
            if retain: return [value(element, True) for _ in range(count)]
            for _ in range(count): value(element, False)
            return None
        if kind not in range(13): raise ValueError("Unknown GGUF value type")
        return unpack({0:"<B", 1:"<b", 2:"<H", 3:"<h", 4:"<I", 5:"<i", 6:"<f", 7:"<?", 10:"<Q", 11:"<q", 12:"<d"}[kind])
    magic, version, tensors, keys = unpack("<4sIQQ")
    if magic != b"GGUF" or version != 3 or tensors != 779 or keys > 4096: raise ValueError("Wrong pinned Flint header")
    metadata = {}
    for _ in range(keys):
        name = string(); retained = name.startswith("flint.") or name.startswith("general.")
        data = value(unpack("<I"), retained)
        if retained: metadata[name] = data
    descriptors = {}
    for _ in range(tensors):
        name, nd = string(), unpack("<I")
        if not 1 <= nd <= 4 or name in descriptors: raise ValueError("Invalid GGUF descriptor")
        dims = [unpack("<Q") for _ in range(nd)]
        kind, offset = unpack("<IQ")
        descriptors[name] = dict(dimensions=dims, type=kind, offset=offset)
    alignment = metadata.get("general.alignment", 32)
    if alignment != 32 or metadata["general.architecture"] != "flint" or metadata["flint.expert_used_count"] != 6:
        raise ValueError("Wrong Flint geometry/alignment")
    return at + (-at % alignment), descriptors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--control", required=True, help="Native flint-weight-control binary, Linux path with --wsl")
    p.add_argument("--wsl", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected = identity(); before = args.model.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if before.st_size != expected["bytes"] or args.model.is_symlink(): raise ValueError("Model size/identity differs")
    started = time.monotonic()
    if hash_prefix(args.model, before.st_size).hexdigest() != expected["sha256"]: raise ValueError("Full model SHA mismatch")
    if stamp(args.model.stat()) != stamp(before): raise ValueError("Model changed during verification")
    data_offset, descriptors = layout(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    cases, bytes_read = [], 0
    def native_path(path):
        resolved = path.resolve()
        return "/mnt/" + resolved.drive[0].lower() + "/" + "/".join(resolved.parts[1:]) if args.wsl else str(resolved)
    with args.model.open("rb") as stream:
        for layer in [2, 26, 48]:
            for family in ["gate", "up", "down"]:
                name = f"blk.{layer}.ffn_{family}_exps.weight"
                descriptor = descriptors[name]; columns, rows, experts = descriptor["dimensions"]
                if descriptor["type"] != 3 or columns % 32 or experts != 256: raise ValueError("Expected canonical Q4_1 expert layout")
                stride = columns // 32 * 20
                for expert in [0, 127, 255]:
                    # Sample different row regions, including the last rows.
                    row = [0, rows // 2, rows - 16][[0, 127, 255].index(expert)]
                    offset = data_offset + descriptor["offset"] + (expert * rows + row) * stride
                    count = 16 * stride
                    if offset + count > before.st_size or count > 2 << 20: raise ValueError("Unbounded expert read")
                    stream.seek(offset); packed = stream.read(count); bytes_read += len(packed)
                    if len(packed) != count: raise ValueError("Truncated expert slice")
                    tile = args.output / "private-weight-slice.bin"
                    with tile.open("xb") as out: out.write(packed)
                    try:
                        command = (["wsl.exe", "-d", "Ubuntu", "--"] if args.wsl else []) + [args.control, native_path(tile), "16", str(columns)]
                        result = subprocess.run(command, capture_output=True, text=True, check=True)
                        native = json.loads(result.stdout)
                    finally: tile.unlink()
                    blocks = np.frombuffer(packed, dtype=np.uint8).reshape(-1, 20)
                    scale = blocks[:, :2].copy().view("<f2").astype(np.float32)
                    bias = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
                    quant = np.concatenate([blocks[:, 4:] & 15, blocks[:, 4:] >> 4], axis=1).astype(np.float32)
                    weights = (scale * quant + bias).reshape(16, columns)
                    actual = np.array(native["weights"], dtype=np.float32).reshape(16, columns)
                    if not np.array_equal(actual, weights): raise ValueError("Canonical dequantization differs from independent layout")
                    x = np.array([(1 if i % 32 < 16 else -1) * (127 - 7 * (i % 16)) for i in range(columns)], dtype=np.float64)
                    reference = weights.astype(np.float64) @ x
                    error = float(np.max(np.abs(np.array(native["dots"]) - reference)))
                    if error > 2e-4: raise ValueError(f"Canonical real-weight dot differs: {error}")
                    cases.append(dict(tensor=name, expert=expert, firstRow=row, rows=16, columns=columns,
                        packedBytesSHA256=hashlib.sha256(packed).hexdigest(), dequantizationExactlyEqual=True,
                        maximumProjectionAbsoluteError=error, trainedGraphExecuted=False, passed=True))
                    print(f"PASS real-weight {name} expert={expert} maxDotError={error:.9g}", flush=True)
    if stamp(args.model.stat()) != stamp(before): raise ValueError("Verified model changed during slice controls")
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'scripts/download_flint_model.py', root / 'Experiments/Flint/Native/weight-control.cpp']
    hash_command = (["wsl.exe", "-d", "Ubuntu", "--"] if args.wsl else []) + ["sha256sum", args.control]
    binary_hash = subprocess.check_output(hash_command, text=True).split()[0]
    if len(binary_hash) != 64 or any(c not in '0123456789abcdef' for c in binary_hash): raise ValueError("Missing native control identity")
    record = dict(schemaVersion=1, model=expected, fullFileSHA256Verified=True, weightSliceBytesRead=bytes_read,
        controlBinarySHA256=binary_hash, sourceHashes={str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        testsPassed=len(cases), cases=cases, maximumProjectionAbsoluteError=max(c["maximumProjectionAbsoluteError"] for c in cases),
        fullTrainedGraphExecuted=False, iPhoneVerified=False, qualityVerified=False)
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(testsPassed=len(cases), weightSliceBytesRead=bytes_read, elapsedSeconds=round(time.monotonic()-started, 2), trainedGraphExecuted=False)))


if __name__ == "__main__": main()
