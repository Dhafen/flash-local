"""Build pinned CPU llama.cpp and dump logits for the generated tiny fixture.

Never accepts a user model/prompt. Actual REAM weights are not fetched or run.
"""
import pathlib
import subprocess

from llama_reference_source import REVISION, prepare


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
            source = prepare(root, run)
            run(["cmake", "-S", root / "Tests/ReferenceHarness", "-B", build,
                 f"-DLLAMA_SOURCE={source}", "-DCMAKE_BUILD_TYPE=Release"])
            run(["cmake", "--build", build, "--target", "flash-local-reference",
                 "flash-local-tokenizer-reference", "--parallel", "3"])
            model = root / ".test-fixtures/synthetic-qwen4exp.gguf"
            if not model.is_file() or model.stat().st_size > 1_000_000:
                raise RuntimeError("Only the tiny synthetic fixture may run in this harness")
            run([build / "flash-local-reference", model, root / ".test-fixtures/llama-reference-logits.json"])
            run(["python3", root / "scripts/make_tokenizer_reference_fixture.py"])
            run([build / "flash-local-tokenizer-reference",
                 root / ".test-fixtures/qwen35-vocabulary-only.gguf",
                 root / ".test-fixtures/qwen35-tokenizer-corpus.bin",
                 root / ".test-fixtures/qwen35-tokenizer-reference.json",
                 root / ".test-fixtures/qwen35-tokenizer-corpus.json"])
        except Exception:
            output.flush()
            print(log.read_text(encoding="utf-8", errors="replace")[-18000:])
            raise
    print(f"Synthetic CPU reference complete at {REVISION}; build log: {log}")


if __name__ == "__main__":
    main()
