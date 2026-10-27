"""Fetch bounded, pinned public tokenizer/header controls, never a full model."""
import argparse
import hashlib
import json
import pathlib
import urllib.request

MODEL_REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
GGUF_REVISION = "9c47151036bc2c34331142dda04488bceab2bc9f"
PREFIX_SHA256 = "a4a3cfd21a3028b7a92057969f47874d0f1dc32f7cf3dd49988dbdcd5ce34390"
FILES = {
    "tokenizer.json": (10_879_128, "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68"),
    "tokenizer_config.json": (30_439, "159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf"),
}


def fetch(request, target, size, sha256, ranged=False):
    if target.exists():
        if target.stat().st_size != size or hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"Existing input differs from pinned source: {target.name}")
        return
    with urllib.request.urlopen(request, timeout=45) as response:
        if ranged and (response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-{size-1}/25992966976"):
            raise ValueError("GGUF server did not honor the exact bounded range; full download refused")
        data = response.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"Pinned metadata digest/size mismatch: {target.name}")
    target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(".work/flint-inputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        url = f"https://huggingface.co/RecursiveSI/flint/resolve/{MODEL_REVISION}/{name}"
        fetch(urllib.request.Request(url), args.output / name, size, digest)
    url = f"https://huggingface.co/RecursiveSI/flint-int4-gguf/resolve/{GGUF_REVISION}/flint-int4-q4_1.gguf"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-8388607", "Accept-Encoding": "identity"})
    fetch(request, args.output / "gguf-prefix.bin", 8_388_608, PREFIX_SHA256, ranged=True)
    print(json.dumps({"downloadedBytes": sum(x[0] for x in FILES.values()) + 8_388_608,
        "fullModelDownloaded": False, "fullModelSHA256Verified": False, "trainedInferenceExecuted": False}))


if __name__ == "__main__": main()
