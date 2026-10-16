"""Pin Ornith's MLX metadata for the isolated QPack experiment.

Downloads config/tokenizer metadata and only the safetensors headers. No
trained tensor payload is requested. Regeneration must use the exact revision;
the generated contract also pins each complete shard's published LFS SHA-256.
"""
import argparse
import base64
import hashlib
import json
import pathlib
import re
import struct
import urllib.request

REPO = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
REVISION = "19504d912fa8fc7622bf6b1de3db5d5d890b1f02"
ENGINE_REVISION = "de1a942241f65ac43290cecffc85d69ea5b52310"
MAX_HEADER = 8 << 20
MAX_AUX = 32 << 20


def fetch(url, maximum, byte_range=None, total=None):
    headers = {"Accept-Encoding": "identity"}
    if byte_range is not None:
        first, end = byte_range
        headers["Range"] = f"bytes={first}-{end - 1}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        if byte_range is not None:
            expected = f"bytes {first}-{end - 1}/{total}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server refused exact bounded metadata range")
        elif response.status != 200:
            raise ValueError("Unexpected metadata response")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Encoded metadata cannot preserve byte ranges")
        data = response.read(maximum + 1)
    if len(data) > maximum or byte_range is not None and len(data) != end - first:
        raise ValueError("Metadata response length differs from its bound")
    return data


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def checked_header(data, file_bytes):
    header = json.loads(data)
    element_bytes = {"F32": 4, "U32": 4, "F16": 2, "BF16": 2}
    tensors = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        shape, dtype, offsets = value["shape"], value["dtype"], value["data_offsets"]
        if not shape or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError(f"Invalid shape for {name}")
        if dtype not in element_bytes or len(offsets) != 2:
            raise ValueError(f"Unsupported tensor storage for {name}")
        size = element_bytes[dtype]
        for dim in shape:
            size *= dim
        if offsets[0] < 0 or offsets[1] - offsets[0] != size:
            raise ValueError(f"Tensor byte count differs from shape for {name}")
        tensors.append((offsets[0], offsets[1], name))
    cursor = 0
    for first, end, name in sorted(tensors):
        if first != cursor:
            raise ValueError(f"Tensor gap or overlap at {name}; lossless reconstruction refused")
        cursor = end
    if 8 + len(data) + cursor != file_bytes:
        raise ValueError("Safetensors payload length differs from the pinned shard")
    return header


def inspect(output):
    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
    metadata = json.loads(fetch(
        f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", 4 << 20))
    if metadata["sha"] != REVISION:
        raise ValueError("Metadata revision differs from the requested immutable revision")
    files = {f["rfilename"]: f for f in metadata["siblings"]}
    contract = dict(schemaVersion=1, repository=REPO, revision=REVISION,
                    swiftletRevision=ENGINE_REVISION, auxiliary=[], shards=[])
    auxiliary_bytes = {}
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json"]:
        if name not in files:
            if name == "generation_config.json":
                continue
            raise ValueError(f"Required tokenizer/config metadata absent: {name}")
        expected = files[name]["size"]
        if expected <= 0 or expected > MAX_AUX:
            raise ValueError("Auxiliary file exceeds the metadata allowance")
        data = fetch(base + name, expected)
        if len(data) != expected:
            raise ValueError(f"Auxiliary length mismatch: {name}")
        auxiliary_bytes[name] = data
        contract["auxiliary"].append(dict(name=name, byteCount=expected, sha256=sha256(data)))
    contract["configBase64"] = base64.b64encode(auxiliary_bytes["config.json"]).decode("ascii")
    weight_map = json.loads(fetch(base + "model.safetensors.index.json", MAX_HEADER))["weight_map"]
    names = set()
    payload_bytes = dense_bytes = expert_bytes = header_bytes = 0
    for name in sorted(set(weight_map.values())):
        if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name):
            raise ValueError("Unexpected shard path")
        entry = files[name]
        file_bytes, digest = entry["size"], entry["lfs"]["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("No published complete-shard SHA-256")
        prefix = fetch(base + name, 8, (0, 8), file_bytes)
        length, = struct.unpack("<Q", prefix)
        if length <= 0 or length > MAX_HEADER:
            raise ValueError("Safetensors header exceeds its bound")
        raw = fetch(base + name, length, (8, 8 + length), file_bytes)
        header = checked_header(raw, file_bytes)
        for tensor, info in header.items():
            if tensor == "__metadata__":
                continue
            if tensor in names or weight_map.get(tensor) != name:
                raise ValueError("Tensor index differs from shard headers")
            names.add(tensor)
            size = info["data_offsets"][1] - info["data_offsets"][0]
            if ".mlp.switch_mlp." in tensor:
                expert_bytes += size
            else:
                dense_bytes += size
        contract["shards"].append(dict(name=name, byteCount=file_bytes, sha256=digest,
            headerSHA256=sha256(raw), headerBase64=base64.b64encode(raw).decode("ascii")))
        header_bytes += len(raw) + 8
        payload_bytes += file_bytes
    if names != set(weight_map):
        raise ValueError("Index has missing or extra tensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(revision=REVISION, tensors=len(names), headerBytes=header_bytes,
                          denseBytes=dense_bytes, expertBytes=expert_bytes,
                          totalSourceBytes=payload_bytes, trainedPayloadDownloaded=False,
                          contractSHA256=sha256(output.read_bytes())), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    inspect(parser.parse_args().output)


if __name__ == "__main__":
    main()
