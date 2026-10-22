"""Portable vocabulary/template oracle; original pinned metadata, zero tensors.

Run on Linux or macOS. Does not create an inference context, load trained
weights, produce a performance number, or replace Apple/native release gates.
"""
import json
from pathlib import Path
import platform
import subprocess

from llama_reference_source import prepare, REVISION


def main():
    root = Path(__file__).resolve().parents[1]
    work = root / ".work"
    work.mkdir(exist_ok=True)
    with (work / "agent-tokenizer-controls.log").open("w", encoding="utf-8") as log:
        def run(arguments):
            subprocess.run([str(value) for value in arguments], cwd=root, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        try:
            source = prepare(root, run)
            build = work / "agent-tokenizer-build"
            configure = ["cmake", "-S", root / "Tests/ReferenceHarness", "-B", build,
                         f"-DLLAMA_SOURCE={source}", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
                         "-DGGML_ACCELERATE=OFF", "-DGGML_CPU_KLEIDIAI=OFF", "-DGGML_LLAMAFILE=OFF"]
            if platform.system() == "Darwin":
                if platform.machine() not in {"arm64", "x86_64"}:
                    raise RuntimeError("Unsupported Apple control architecture")
                configure += [f"-DCMAKE_OSX_ARCHITECTURES={platform.machine()}"]
            run(configure)
            run(["cmake", "--build", build, "--target", "flash-local-tokenizer-reference", "--parallel", "3"])
            for name in ["ream", "ornith"]:
                run(["python3", root / "scripts" / f"fetch_{name}_metadata.py"])
                run(["python3", root / "scripts/make_tokenizer_reference_fixture.py", *(["--ornith"] if name == "ornith" else [])])
                folder = root / ".test-fixtures" / ("ornith-tokenizer" if name == "ornith" else "")
                run([build / "flash-local-tokenizer-reference", folder / "qwen35-vocabulary-only.gguf",
                     folder / "qwen35-tokenizer-corpus.bin", folder / "qwen35-tokenizer-reference.json",
                     folder / "qwen35-tokenizer-corpus.json"])
        except Exception:
            log.flush()
            print((work / "agent-tokenizer-controls.log").read_text(errors="replace")[-9000:])
            raise
    text = (work / "agent-tokenizer-controls.log").read_text()
    marker = "CANONICAL_AGENT_BRIDGE cases=4 invalidCases=8 exactRender=true bothThinkingModes=true bufferCanaries=true metadataOnly=true trainedInference=false"
    if text.count(marker) != 2:
        raise RuntimeError("Both original template oracles must complete without skips")
    manifest = {"sourceRevision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "tokenizerRevision": REVISION, "templates": ["ream", "ornith"], "trainedInference": False,
                "replacesAppleNativeGates": False, "platform": platform.system()}
    (work / "agent-tokenizer-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(marker + " templates=2")


if __name__ == "__main__":
    main()
