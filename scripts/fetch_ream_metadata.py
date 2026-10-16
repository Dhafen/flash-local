"""Fetch a pinned GGUF header, never model payload. Abort on unbounded HTTP."""
import hashlib
import pathlib
import urllib.request

REVISION = "28a55fecdca51a498900aeb34a4e7e8338977cd3"
NAME = "Qwen3.8-Flash-Next-REAM-60Pct-HQ-PLE4of16-56GiB.gguf"
HEADER_BYTES = 11_023_104
MODEL_BYTES = 56_115_609_536
HEADER_SHA256 = "352b87fd036c3b14c4bcb3ff83c19dc1802be26c93686b4300972e9a43390292"
URL = ("https://huggingface.co/RemySkye/Qwen3.8-Flash-Next-REAM-60Pct-GGUF/resolve/"
       + REVISION + "/" + NAME)


def main():
    request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = f"bytes 0-{HEADER_BYTES - 1}/{MODEL_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError("Server did not honor the exact bounded metadata request")
        data = response.read(HEADER_BYTES + 1)
    if len(data) != HEADER_BYTES or hashlib.sha256(data).hexdigest() != HEADER_SHA256:
        raise RuntimeError("Pinned GGUF header length/hash mismatch")
    folder = pathlib.Path(__file__).resolve().parents[1] / ".test-fixtures"
    folder.mkdir(exist_ok=True)
    output = folder / "ream.header"
    output.write_bytes(data)
    print(f"Verified {len(data)} header bytes; no model payload downloaded: {output}")


if __name__ == "__main__":
    main()
