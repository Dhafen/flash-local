"""Resumable single-copy download of the pinned public Flint GGUF.

Bounded 4 MiB memory, SHA-256 before publication, no inference or quantization.
The caller selects a cache directory with enough free space. Never replaces an
unverified existing model/partial belonging to a different download identity.
"""
import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
import pathlib
import shutil
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILE = ROOT / "Experiments/Flint/model-profile.json"
CHUNK = 4 << 20


class TransferInterrupted(Exception): pass


@contextmanager
def exclusive_download(directory):
    path = directory / ".flint-download.lock"
    if path.is_symlink(): raise ValueError("Download lock must be a local regular file")
    with path.open("a+b") as lock:
        if lock.seek(0, os.SEEK_END) == 0: lock.write(b"0"); lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            lock.seek(0)
            if os.name == "nt": msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def file_identity(path):
    stat = path.stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size, mtimeNS=stat.st_mtime_ns, ctimeNS=stat.st_ctime_ns)


def emit(**values):
    print(json.dumps(values), flush=True)


def identity():
    artifact = json.loads(PROFILE.read_text(encoding="utf-8"))["artifact"]
    return {"schemaVersion": 1, "repository": artifact["repository"], "revision": artifact["revision"],
            "fileName": artifact["fileName"], "bytes": artifact["bytes"], "sha256": artifact["sha256"]}


def hash_prefix(path, count):
    digest, read, last = hashlib.sha256(), 0, time.monotonic()
    with path.open("rb") as stream:
        while read < count:
            data = stream.read(min(CHUNK, count - read))
            if not data: raise ValueError("Download prefix was truncated")
            digest.update(data); read += len(data)
            if time.monotonic() - last >= 15:
                emit(phase="checking-existing-bytes", bytes=read, total=count)
                last = time.monotonic()
    return digest


def fetch(directory, retries=6):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_download(directory): return fetch_locked(directory, retries)


def fetch_locked(directory, retries):
    model = identity()
    final = directory / model["fileName"]
    partial = directory / (model["fileName"] + ".partial")
    seal = directory / "flint-download.json"
    receipt = directory / "flint-verified.json"
    def save_receipt():
        result = {**model, "fullFileSHA256Verified": True, "verifiedAtUnix": int(time.time()),
                  "fileIdentity": file_identity(final), "trainedInferenceExecuted": False}
        temporary = directory / "flint-verified.json.new"
        if temporary.is_symlink(): raise ValueError("Receipt must not be a symlink")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt)
    if final.exists():
        if not final.is_file() or final.is_symlink() or final.stat().st_size != model["bytes"]:
            raise ValueError("Existing model differs from the pinned file; not replaced")
        before = file_identity(final)
        if hash_prefix(final, model["bytes"]).hexdigest() != model["sha256"] or before != file_identity(final):
            raise ValueError("Existing full model SHA-256 mismatch; not replaced")
        save_receipt()
        emit(phase="verified", bytes=model["bytes"], sha256=model["sha256"], reused=True)
        return final
    if partial.exists():
        if not partial.is_file() or partial.is_symlink() or not seal.is_file() or json.loads(seal.read_text()) != model:
            raise ValueError("Existing partial download has no matching pinned identity")
    elif seal.exists():
        if json.loads(seal.read_text()) != model: raise ValueError("Existing download identity differs")
    else:
        with seal.open("x", encoding="utf-8") as out: json.dump(model, out, indent=2)
    position = partial.stat().st_size if partial.exists() else 0
    if position > model["bytes"]: raise ValueError("Partial file exceeds the model size")
    if shutil.disk_usage(directory).free < model["bytes"] - position + (10 << 30):
        raise ValueError("Cache volume needs the remaining model bytes plus 10 GiB free reserve")
    digest = hash_prefix(partial, position) if position else hashlib.sha256()
    started, original, last, failures = time.monotonic(), position, time.monotonic(), 0
    url = f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/{model["fileName"]}'
    while position < model["bytes"]:
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "Range": f'bytes={position}-{model["bytes"]-1}'})
        try:
            try: response = urllib.request.urlopen(request, timeout=45)
            except (OSError, urllib.error.URLError) as error: raise TransferInterrupted() from error
            with response:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Compressed response is not an exact GGUF range")
                if response.status == 206:
                    expected = f'bytes {position}-{model["bytes"]-1}/{model["bytes"]}'
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("Server returned a different range/file size")
                elif response.status == 200 and position == 0:
                    if int(response.headers.get("Content-Length", -1)) != model["bytes"]:
                        raise ValueError("Server full-file length disagrees with pinned identity")
                else: raise ValueError("Server did not honor a resumable model request")
                with partial.open("ab") as output:
                    if os.fstat(output.fileno()).st_size != position:
                        raise ValueError("Partial file changed during transfer")
                    while position < model["bytes"]:
                        try: data = response.read(min(CHUNK, model["bytes"] - position))
                        except (OSError, http.client.IncompleteRead) as error: raise TransferInterrupted() from error
                        if not data: raise TransferInterrupted("Model response ended early")
                        if len(data) > min(CHUNK, model["bytes"] - position): raise ValueError("Oversized model response")
                        output.write(data); output.flush()
                        digest.update(data); position += len(data)
                        if time.monotonic() - last >= 15:
                            rate = (position - original) / max(.001, time.monotonic() - started)
                            emit(phase="downloading", bytes=position, total=model["bytes"], percent=round(100 * position / model["bytes"], 2), bytesPerSecond=round(rate))
                            last = time.monotonic()
                    output.flush(); os.fsync(output.fileno())
            failures = 0
        except TransferInterrupted as error:
            failures += 1
            if failures > retries: raise RuntimeError("Model transfer paused after repeated network errors; rerun to resume") from error
            emit(phase="retry", bytes=position, attempt=failures, errorType=type(error).__name__)
            time.sleep(min(30, 2 ** failures))
    if digest.hexdigest() != model["sha256"]:
        raise ValueError("Full downloaded model SHA-256 mismatch; partial retained and not admitted")
    # Atomic no-clobber publication with no second payload copy on NTFS/POSIX.
    # If hard links are unavailable, retain the verified partial and fail.
    os.link(partial, final, follow_symlinks=False)
    partial.unlink()
    save_receipt()
    emit(phase="verified", bytes=model["bytes"], sha256=model["sha256"], elapsedSeconds=round(time.monotonic() - started, 2))
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    fetch(args.output)
