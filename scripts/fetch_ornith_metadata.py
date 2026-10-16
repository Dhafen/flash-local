"""Fetch only the exact pinned Ornith Q4 metadata/index, never weight bodies."""
import hashlib
import pathlib
import urllib.request

REVISION = "12393612fd4f730ff5aadc23e9b8f9648aa49ceb"
NAME = "Ornith-1.5-35B-Q4_K_M.gguf"
HEADER_BYTES = 10_990_848
MODEL_BYTES = 21_713_463_040
HEADER_SHA256 = "961df6ceaeedc2fbc0c84dfc16d8ba3ebf1d7e67506d10096a4d3a52d8a0836b"
URL = ("https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/"
       + REVISION + "/" + NAME)


def main():
    request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-{HEADER_BYTES - 1}/{MODEL_BYTES}":
            raise RuntimeError("Ornith server refused the exact bounded metadata request")
        header = response.read(HEADER_BYTES + 1)
    if len(header) != HEADER_BYTES or hashlib.sha256(header).hexdigest() != HEADER_SHA256:
        raise RuntimeError("Pinned Ornith header length/hash mismatch")
    folder = pathlib.Path(__file__).resolve().parents[1] / ".test-fixtures"
    folder.mkdir(exist_ok=True)
    (folder / "ornith.header").write_bytes(header)
    print(f"ORNITH_HEADER bytes={len(header)} exact206=true hashVerified=true tensorPayload=false")


if __name__ == "__main__":
    main()
