"""Public fixed-prompt tokenizer oracle; never downloads or runs model weights."""
import argparse
import hashlib
import json
import pathlib

from inspect_ornith_mlx import fetch, REPO, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    source = json.loads((root / "Experiments/OrnithPerformance/Sources/OrnithPerformance/Resources/ornith-mlx-source.json").read_text())
    assert source["repository"] == REPO and source["revision"] == REVISION
    args.output.mkdir(parents=True, exist_ok=True)
    for file in source["auxiliary"]:
        if file["name"] not in {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"}:
            raise ValueError("Weight or unknown file requested by tokenizer oracle")
        path = args.output / file["name"]
        data = path.read_bytes() if path.is_file() else b""
        if len(data) != file["byteCount"] or hashlib.sha256(data).hexdigest() != file["sha256"]:
            data = fetch(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{file['name']}", file["byteCount"])
        if len(data) != file["byteCount"] or hashlib.sha256(data).hexdigest() != file["sha256"]:
            raise ValueError("Pinned tokenizer metadata hash mismatch")
        path.write_bytes(data)
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.output, local_files_only=True, trust_remote_code=False)
    template = (args.output / "chat_template.jinja").read_text(encoding="utf-8")
    conversations = [
        [{"role": "user", "content": "Hej"}],
        [{"role": "user", "content": "Vad är 50+5?"}],
        [{"role": "user", "content": "Write a Python function that sorts a list without mutating its input."}],
        [{"role": "user", "content": "åäö 🙂 漢字 e\u0301\n```swift\nlet x = 42\n```"}],
        [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "What is 6 times 7?"}],
        [{"role": "user", "content": "Hej"}, {"role": "assistant", "content": "Hej!"}, {"role": "user", "content": "Vad är 50+5?"}],
    ]
    cases = []
    for messages in conversations:
        for thinking in [False, True]:
            ids = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=True,
                add_generation_prompt=True, enable_thinking=thinking, return_dict=False)
            if not isinstance(ids, list) or not all(type(token) is int for token in ids):
                raise ValueError("Tokenizer oracle did not return a flat token-ID list")
            cases.append(dict(messages=messages, thinking=thinking, tokenIDs=ids,
                              decoded=tokenizer.decode(ids, skip_special_tokens=False)))
    result = dict(repository=REPO, revision=REVISION, transformersVersion=transformers.__version__,
                  tokenizersVersion=tokenizers.__version__, trainedInference=False, cases=cases)
    (args.output / "reference.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ORNITH_MLX_TOKENIZER_REFERENCE cases={len(cases)} exactPinnedMetadata=true trainedWeightDownload=false trainedInference=false")


if __name__ == "__main__":
    main()
