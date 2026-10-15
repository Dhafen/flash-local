"""The single source pin shared by the runtime and numerical reference."""
import pathlib
import subprocess

REVISION = "e467e5b5e5ba43071d6151b3d881deb9300d000e"
REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"


def prepare(root: pathlib.Path, run):
    source = root / ".work/llama-reference"
    source.parent.mkdir(exist_ok=True)
    if not source.exists():
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "--no-checkout", REPOSITORY, source])
    if not (source / ".git").is_dir():
        raise RuntimeError("Reference cache is not a Git checkout")
    if (source / ".git/index").exists():
        run(["git", "-C", source, "diff", "--exit-code"])
        run(["git", "-C", source, "diff", "--cached", "--exit-code"])
    unknown = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--others"], text=True)
    if unknown.strip():
        raise RuntimeError("Untracked files could alter the pinned source build")
    run(["git", "-C", source, "sparse-checkout", "set", "src", "include", "ggml", "cmake", "vendor", "common"])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        run(["git", "-C", source, "fetch", "--depth", "1", "origin", REVISION])
        run(["git", "-C", source, "checkout", "--detach", REVISION])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError("Reference revision mismatch")
    return source
