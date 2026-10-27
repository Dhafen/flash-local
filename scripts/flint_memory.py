"""Conservative admission estimates, never substitute for physical iPhone tests."""
import json
import pathlib

PROFILE = pathlib.Path(__file__).resolve().parents[1] / "Experiments/Flint/model-profile.json"


def plan(context, physical_batch=4, process_ceiling=5_500_000_000, app_bytes=400_000_000,
         scratch_reserve=1_000_000_000, available_bytes=None):
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    g, layout = profile["geometry"], profile["layoutAudit"]
    if isinstance(context, bool) or not isinstance(context, int) or not 1 <= context <= g["metadataContextLimit"]:
        raise ValueError("Context exceeds the model metadata")
    if isinstance(physical_batch, bool) or not isinstance(physical_batch, int) or not 1 <= physical_batch <= 256:
        raise ValueError("Unsupported physical batch")
    for size in [process_ceiling, app_bytes, scratch_reserve]:
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size < (1 << 63): raise ValueError("Invalid memory budget")
    if available_bytes is not None and (not isinstance(available_bytes, int) or isinstance(available_bytes, bool) or available_bytes < 0):
        raise ValueError("Invalid available-memory observation")
    # Native KV allocation is padded. Include a microbatch's possible live
    # SWA tokens, round each per-layer capacity to 256, and do not call this a
    # measured footprint. No extra full-KV prefix snapshot is budgeted/allowed.
    align = lambda n: (n + 255) // 256 * 256
    full = len(g["fullAttentionWithoutRoPE"])
    swa = g["layers"] - full
    kv = (full * align(context) + swa * align(min(context, g["slidingWindow"] + physical_batch))) * g["kvHeads"] * g["headDimension"] * 2 * 2
    slots = min(g["routedExperts"], g["topK"] * physical_batch)
    bank = slots * (layout["routedBytes"] // (g["layers"] - g["leadingDenseLayers"]) // g["routedExperts"])
    dense = layout["nonRoutedBytes"]
    required = app_bytes + dense + kv + bank + scratch_reserve
    limit = process_ceiling
    if available_bytes is not None: limit = min(limit, app_bytes + available_bytes)
    return {"estimated": True, "contextTokens": context, "physicalBatch": physical_batch,
            "residentDenseUpperBoundBytes": dense, "paddedKVBytes": kv, "expertSlots": slots,
            "expertBankBytes": bank, "appBytes": app_bytes, "scratchReserveBytes": scratch_reserve,
            "estimatedProcessBytes": required, "effectiveCeilingBytes": limit,
            "admitted": required <= limit, "iPhoneStabilityVerified": False}


if __name__ == "__main__":
    print(json.dumps([plan(n) for n in [8192, 32768, 65536]], indent=2))
