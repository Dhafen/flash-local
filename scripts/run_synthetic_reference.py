"""Build pinned CPU llama.cpp and dump logits for the generated tiny fixture.

Never accepts a user model/prompt. Actual REAM weights are not fetched or run.
"""
import pathlib
import hashlib
import json
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
            command = [str(a) for a in args]
            try:
                subprocess.run(command, cwd=root, stdout=output,
                               stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                if error.returncode < 0 and pathlib.Path(command[0]).name == "flash-local-mixed-fixture":
                    # One diagnostic rerun of our bounded random fixture only.
                    # Keep the original failure, even if the debugger run exits.
                    subprocess.run(["lldb", "--batch", "-o", "run", "-o", "thread backtrace all", "--"] + command,
                                   cwd=root, stdout=output, stderr=subprocess.STDOUT, timeout=60)
                raise
        try:
            source = prepare(root, run)
            run(["cmake", "-S", root / "Tests/ReferenceHarness", "-B", build,
                 f"-DLLAMA_SOURCE={source}", "-DCMAKE_BUILD_TYPE=Release"])
            run(["cmake", "--build", build, "--target", "flash-local-reference",
                 "flash-local-tokenizer-reference", "flash-local-session-reference",
                 "flash-local-mixed-fixture", "--parallel", "3"])
            model = root / ".test-fixtures/synthetic-qwen4exp.gguf"
            if not model.is_file() or model.stat().st_size > 1_000_000:
                raise RuntimeError("Only the tiny synthetic fixture may run in this harness")
            run([build / "flash-local-reference", model, root / ".test-fixtures/llama-reference-logits.json"])
            run(["python3", root / "scripts/make_qwen4exp_test_model.py", "--mixed"])
            mixed = root / ".test-fixtures/synthetic-qwen4exp-mixed.gguf"
            run([build / "flash-local-mixed-fixture",
                 root / ".test-fixtures/synthetic-qwen4exp-mixed-source.gguf", mixed])
            if mixed.stat().st_size > 32 << 20:
                raise RuntimeError("Mixed reference is restricted to the generated random fixture")
            manifest = json.loads((root / ".test-fixtures/synthetic-qwen4exp-mixed-source.json").read_text())
            manifest.update(fileName=mixed.name, fileByteCount=mixed.stat().st_size,
                            sha256=hashlib.sha256(mixed.read_bytes()).hexdigest())
            (root / ".test-fixtures/synthetic-qwen4exp-mixed.json").write_text(json.dumps(manifest, indent=2) + "\n")
            run([build / "flash-local-reference", mixed,
                 root / ".test-fixtures/llama-mixed-reference-logits.json"])
            run(["python3", root / "scripts/make_tokenizer_reference_fixture.py"])
            run([build / "flash-local-tokenizer-reference",
                 root / ".test-fixtures/qwen35-vocabulary-only.gguf",
                 root / ".test-fixtures/qwen35-tokenizer-corpus.bin",
                 root / ".test-fixtures/qwen35-tokenizer-reference.json",
                 root / ".test-fixtures/qwen35-tokenizer-corpus.json"])
            run(["python3", root / "scripts/make_qwen4exp_test_model.py", "--chat"])
            chat = root / ".test-fixtures/synthetic-qwen4exp-chat.gguf"
            if not chat.is_file() or chat.stat().st_size > 1_000_000:
                raise RuntimeError("Only the tiny random-weight chat fixture may run")
            run([build / "flash-local-session-reference", chat,
                 root / ".test-fixtures/synthetic-qwen4exp-chat.json",
                 root / ".test-fixtures/qwen4exp-session-reference.json"])
        except Exception:
            output.flush()
            print(log.read_text(encoding="utf-8", errors="replace")[-18000:])
            raise
    print(f"Synthetic CPU reference complete at {REVISION}; build log: {log}")


if __name__ == "__main__":
    main()
