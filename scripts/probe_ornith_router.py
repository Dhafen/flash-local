"""Run a bounded, gradient-free perturbation of one real Ornith router tensor.

Hidden states are synthetic. This probe does not measure model behavior,
quality, routing locality on text, or performance on an iPhone.
"""
import argparse
import hashlib
import json
from pathlib import Path

from inspect_ornith_mlx import fetch

REPO = "ornith-ai/Ornith-1.5-35B-A3B"
REVISION = "10fbf86fed7ecee4a061f8b499a618f46001cac1"
SHARD = "model-00001-of-00016.safetensors"
SHARD_BYTES = 4_744_232_576
SHARD_PUBLISHED_SHA256 = "250b3b09199be0dcd0f96df2a35dabad07e2ba709bb7687ef7490d0f248eb94e"
TENSOR = "model.language_model.layers.0.mlp.gate.weight"
TENSOR_RANGE = (2_695_214_880, 2_696_263_456)
TENSOR_SHA256 = "455f761a199a64794c7daa78ddbe7e279f11cdab8d7bafa1d0b612c1d692a107"
SHAPE = (256, 2048)
TOP_K = 8
GAIN = 0.5


def run(payload):
    if hashlib.sha256(payload).hexdigest() != TENSOR_SHA256:
        raise ValueError("The router payload does not match the pinned tensor SHA-256")
    import numpy as np

    def top_k(weights, x):
        logits = x @ weights.T
        return np.argsort(-logits, axis=-1, kind="stable")[:, :TOP_K]

    def bf16_round(values):
        bits = values.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
        return (rounded & np.uint32(0xFFFF0000)).view(np.float32)

    original = (np.frombuffer(payload, dtype="<u2").astype(np.uint32) << np.uint32(16))
    original = original.view(np.float32).reshape(SHAPE)
    if not np.isfinite(original).all():
        raise ValueError("Non-finite original weight")
    original.setflags(write=False)
    train_seed, holdout_seed = 6021, 6022
    training = np.random.default_rng(train_seed).standard_normal((32, SHAPE[1]), dtype=np.float32)
    holdout = np.random.default_rng(holdout_seed).standard_normal((128, SHAPE[1]), dtype=np.float32)
    base_training = top_k(original, training)
    expert = int(np.bincount(base_training.ravel(), minlength=SHAPE[0]).argmax())
    direction = training.mean(axis=0)
    direction /= np.linalg.norm(direction)
    patched = original.copy()
    patched[expert] = bf16_round(patched[expert] + GAIN * direction)

    def compare(states):
        before, after = top_k(original, states), top_k(patched, states)
        changed = TOP_K - (before[:, :, None] == after[:, None, :]).any(axis=-1).sum(axis=-1)
        return {
            "samples": len(states),
            "targetBefore": int((before == expert).any(axis=-1).sum()),
            "targetAfter": int((after == expert).any(axis=-1).sum()),
            "samplesWithChangedExpertSet": int((changed > 0).sum()),
            "expertSelectionsReplaced": int(changed.sum()),
        }

    result = {
        "repository": REPO, "revision": REVISION, "tensor": TENSOR, "shape": SHAPE,
        "shard": SHARD, "publishedShardSHA256": SHARD_PUBLISHED_SHA256,
        "verifiedTensorSHA256": TENSOR_SHA256, "tensorBytes": len(payload),
        "syntheticHiddenStates": True, "fullModelExecuted": False,
        "patch": {"expert": expert, "gain": GAIN, "trainSeed": train_seed, "holdoutSeed": holdout_seed,
                  "changedBF16Weights": int(np.count_nonzero(patched[expert] != original[expert]))},
        "training": compare(training), "holdout": compare(holdout),
    }
    patched[expert] = original[expert]
    result["rollbackExact"] = bool(np.array_equal(patched.view(np.uint32), original.view(np.uint32)))
    if not result["rollbackExact"]:
        raise RuntimeError("Router rollback failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-file", type=Path, help="Read the pinned 1 MiB BF16 range from a local file")
    args = parser.parse_args()
    count = TENSOR_RANGE[1] - TENSOR_RANGE[0]
    if args.tensor_file:
        with args.tensor_file.open("rb") as stream:
            payload = stream.read(count + 1)
    else:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{SHARD}"
        payload = fetch(url, count, TENSOR_RANGE, SHARD_BYTES)
    print(json.dumps(run(payload), indent=2))


if __name__ == "__main__":
    main()
