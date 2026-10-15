"""Build pinned CPU llama.cpp and dump logits for the generated tiny fixture.

Never accepts a user model/prompt. Actual REAM weights are not fetched or run.
"""
import pathlib
import subprocess

REVISION = "e467e5b5e5ba43071d6151b3d881deb9300d000e"
REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    work = root / ".work"
    work.mkdir(exist_ok=True)
    source, build = work / "llama-reference", work / "reference-build"
    log = work / "reference-build.log"
    with log.open("w", encoding="utf-8") as output:
        def run(args):
            subprocess.run([str(a) for a in args], cwd=root, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        try:
            if not source.exists():
                run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                     "--no-checkout", REPOSITORY, source])
            if not (source / ".git").is_dir():
                raise RuntimeError("Reference cache is not a Git checkout")
            run(["git", "-C", source, "diff", "--exit-code"])
            run(["git", "-C", source, "sparse-checkout", "set", "src", "include", "ggml", "cmake", "vendor"])
            run(["git", "-C", source, "fetch", "--depth", "1", "origin", REVISION])
            run(["git", "-C", source, "checkout", "--detach", REVISION])
            actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if actual != REVISION:
                raise RuntimeError("Reference revision mismatch")
            run(["cmake", "-S", root / "Tests/ReferenceHarness", "-B", build,
                 f"-DLLAMA_SOURCE={source}", "-DCMAKE_BUILD_TYPE=Release"])
            run(["cmake", "--build", build, "--target", "flash-local-reference", "--parallel", "3"])
            model = root / ".test-fixtures/synthetic-qwen4exp.gguf"
            if not model.is_file() or model.stat().st_size > 1_000_000:
                raise RuntimeError("Only the tiny synthetic fixture may run in this harness")
            run([build / "flash-local-reference", model, root / ".test-fixtures/llama-reference-logits.json"])
        except Exception:
            output.flush()
            print(log.read_text(encoding="utf-8", errors="replace")[-18000:])
            raise
    print(f"Synthetic CPU reference complete at {REVISION}; build log: {log}")


if __name__ == "__main__":
    main()
