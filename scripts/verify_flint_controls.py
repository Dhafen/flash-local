"""Run real native SMALL-model graph controls, not trained phone benchmarks."""
import argparse
import json
import pathlib
import subprocess
import hashlib
import re
import numpy as np
from make_flint_test_model import make
from flint_reference import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--wsl")
    parser.add_argument("--quantizer", help="Canonical small-fixture GGML encoder")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(".work/flint-control-results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    def path(p):
        text = str(p.resolve())
        return "/mnt/" + text[0].lower() + text[2:].replace("\\", "/") if args.wsl else text
    def run(command, **kwargs):
        if args.wsl: command = ["wsl.exe", "-d", args.wsl, "--"] + command
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, **kwargs)
    for name, amplification, experts in [("normal", 1.0, 16), ("high-activation", 64.0, 16), ("wide-router", 1.0, 256)]:
        directory = args.output / name
        identity = make(directory, amplification, expert_count=experts)
        model = directory / "synthetic-flint-f32.gguf"
        tokens = [1 + i % 31 for i in range(40)]
        reference, reference_routes = evaluate(model, tokens)

        def native(physical, streaming, steps=40):
            command = [args.control, path(model), path(directory / "manifest.json") if streaming else "-", str(steps), str(physical)]
            result = run(command)
            label = f"{'stream' if streaming else 'resident'}-{physical}-{steps}"
            (directory / f"{label}.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode: raise RuntimeError(f"{name}/{label} failed: {result.stderr[-2500:]}")
            out = json.loads(result.stdout)
            (directory / f"{label}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
            logits = np.array(out.pop("logits"))
            actual_routes = sorted(out.pop("routes"), key=lambda x: (x["position"], x["layer"]))
            expected_logits, expected_routes = (reference, reference_routes) if steps == 40 else evaluate(model, [1 + i % 31 for i in range(steps)])
            if actual_routes != expected_routes: raise RuntimeError(f"{name}/{label} top-6 routing mismatch")
            maximum_expert = max(i for row in actual_routes for i in row["experts"])
            if experts == 256 and maximum_expert < 128: raise RuntimeError("Wide-router control did not exercise high expert IDs")
            maximum = float(np.max(np.abs(logits - expected_logits)))
            rms = float(np.sqrt(np.mean((logits - expected_logits)**2)))
            same_top1 = bool(np.array_equal(logits.argmax(axis=-1), expected_logits.argmax(axis=-1)))
            if maximum > 2e-4 or not same_top1: raise RuntimeError(f"{name}/{label} logit mismatch: max={maximum} rms={rms} top1={same_top1}")
            if streaming and out["ioBytes"] == 0: raise RuntimeError("Streaming control performed no reads")
            ring = re.search(r"creating\s+SWA KV cache, size =\s*(\d+) cells", result.stderr)
            rotated = bool(ring and steps > int(ring[1]))
            if steps == 300 and not rotated: raise RuntimeError("Control did not exceed actual SWA ring capacity")
            cases.append({"fixture": name, **out, "reference": "independent-numpy-fp64", "swaRingRolloverExercised": rotated,
                          "expertCount": experts, "maximumSelectedExpertID": maximum_expert,
                          "routingDecisions": len(actual_routes) * 6, "exactRouting": True, "maximumAbsoluteError": maximum,
                          "rmsError": rms, "allTop1Equal": same_top1, "passed": True})
            print(f"PASS {name}/{label}: maxLogitError={maximum:.9g}", flush=True)
        for physical, streaming in [(1, False), (4, False), (1, True), (2, True), (4, True), (8, True)]: native(physical, streaming)
        if name == "normal":
            for physical, streaming in [(1, False), (4, True), (8, True)]: native(physical, streaming, steps=300)
        if args.quantizer:
            quantized = directory / "q4_1"
            encoded = run([args.quantizer, path(model), path(quantized)])
            if encoded.returncode: raise RuntimeError(encoded.stderr)
            qt = quantized / "synthetic-flint-q4_1.gguf"
            expected_logits, expected_routes = None, None
            for physical, streaming in [(1, False), (4, False), (1, True), (2, True), (4, True), (8, True)]:
                label = f"{'stream' if streaming else 'resident'}-{physical}"
                result = run([args.control, path(qt), path(quantized / "manifest.json") if streaming else "-", "40", str(physical)])
                (quantized / f"{label}.log").write_text(result.stderr, encoding="utf-8")
                if result.returncode: raise RuntimeError(result.stderr[-2000:])
                out = json.loads(result.stdout)
                actual_logits = np.array(out.pop("logits"))
                actual_routes = sorted(out.pop("routes"), key=lambda x: (x["position"], x["layer"]))
                maximum_expert = max(i for row in actual_routes for i in row["experts"])
                if experts == 256 and maximum_expert < 128: raise RuntimeError("Q4_1 wide routing missed high IDs")
                if expected_logits is None: expected_logits, expected_routes = actual_logits, actual_routes
                maximum = float(np.max(np.abs(actual_logits - expected_logits)))
                if maximum > 2e-5 or actual_routes != expected_routes: raise RuntimeError(f"{name}/Q4_1/{label} changed logits/routes")
                if streaming and out["ioBytes"] == 0: raise RuntimeError("Q4_1 stream performed no reads")
                cases.append({"fixture": name + "-q4_1", **out, "reference": "canonical-q4_1-resident-cpu", "expertCount": experts,
                    "maximumSelectedExpertID": maximum_expert,
                    "maximumAbsoluteError": maximum, "exactRouting": True, "passed": True})
                print(f"PASS {name}/Q4_1/{label}: maxLogitError={maximum:.9g}", flush=True)
            # The portable F32-activation reference is independently checked
            # against the full NumPy/FP64 graph, not against GGML's Q8_1 CPU
            # activations. It must preserve actual routing and top-1. The
            # ordinary CPU controls above continue to exercise unchanged math.
            for physical, streaming, steps in [(1, False, 40), (1, True, 40), (4, True, 40),
                                               (8, True, 40)] + ([(4, True, 300)] if name == "normal" else []):
                label = f"{'stream' if streaming else 'resident'}-{physical}-{steps}"
                result = run([args.control, path(qt), path(quantized / "manifest.json") if streaming else "-",
                              str(steps), str(physical), "q4-f32-reference"])
                (quantized / f"f32-reference-{label}.log").write_text(result.stderr, encoding="utf-8")
                if result.returncode: raise RuntimeError(f"{name}/F32-reference/{label}: {result.stderr[-2000:]}")
                out = json.loads(result.stdout)
                if out.get("q4F32Reference") is not True:
                    raise RuntimeError("Diagnostic did not select the Q4_1/F32 reference")
                actual = np.asarray(out.pop("logits"), dtype=np.float64)
                actual_routes = sorted(out.pop("routes"), key=lambda x: (x["position"], x["layer"]))
                oracle, oracle_routes = evaluate(qt, [1 + i % 31 for i in range(steps)])
                oracle_routes = sorted(oracle_routes, key=lambda x: (x["position"], x["layer"]))
                maximum = float(np.max(np.abs(actual - oracle)))
                rms = float(np.sqrt(np.mean((actual - oracle) ** 2)))
                same_top1 = bool(np.array_equal(actual.argmax(axis=-1), oracle.argmax(axis=-1)))
                if maximum > 2e-4 or not same_top1 or actual_routes != oracle_routes:
                    raise RuntimeError(f"{name}/F32-reference/{label} differs from the FP64 oracle: max={maximum} rms={rms}")
                if streaming and out["ioBytes"] == 0: raise RuntimeError("Q4_1 F32 reference performed no streaming reads")
                cases.append({"fixture": name + "-q4_1-f32-reference", **out, "expertCount": experts,
                    "reference": "independent-q4_1-f32-numpy-fp64", "maximumAbsoluteError": maximum,
                    "rmsError": rms, "allTop1Equal": same_top1, "exactRouting": True, "passed": True})
                print(f"PASS {name}/Q4_1-F32/{label}: maxLogitError={maximum:.9g}", flush=True)
    root = pathlib.Path(__file__).resolve().parents[1]
    native = root / "Experiments/Flint/Native"
    source = json.loads((root / ".work/flint-source/flash-local-flint-source.json").read_text(encoding="utf-8"))
    report = {"schemaVersion": 1, "syntheticOnly": True, "trainedModelVerified": False, "iPhoneVerified": False,
        "baseRuntimeRevision": source["baseRevision"], "upstreamFlintRevision": source["flintGraphRevision"],
        "patchedSources": source["patches"],
        "nativeFiles": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native.iterdir() if p.is_file()},
        "cases": cases}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
