"""Estimate the native demand read ledger on a verified Ornith route trace.

Replay uses one persistent context and an empty initial expert cache. It models
original tensor-major GGUF spans, not physical storage or model compute.
"""
import argparse
from collections import OrderedDict
from itertools import groupby
import json
from pathlib import Path


def span_count(experts):
    # Match the runtime's bounded iovec width. Never include a gap.
    spans = width = 0
    previous = -2
    for expert in sorted(experts):
        if expert != previous + 1 or width == 8:
            spans += 1
            width = 0
        width += 1
        previous = expert
    return spans


def replay(meta, events, physical, width, retain):
    experts_by_layer = meta["expert_bytes"]
    windows = [OrderedDict() for _ in experts_by_layer]
    was_window = False
    ledger = {phase: {"expertMisses": 0, "bytes": 0, "logicalReadSpans": 0}
              for phase in ("prefill", "decode")}
    for _, sequence in groupby(events, key=lambda e: e["sequence_id"]):
        for phase, entries in groupby(sequence, key=lambda e: e["phase"]):
            batch_size = physical if phase == "prefill" else 1
            for _, batch in groupby(entries, key=lambda e: e["token_index"] // batch_size):
                requested = {}
                token_positions = set()
                for event in batch:
                    token_positions.add(event["token_index"])
                    row = requested.setdefault(event["layer"], OrderedDict())
                    for expert in event["experts"]:
                        row[expert] = None
                        row.move_to_end(expert)
                window = width > 0 and len(token_positions) == 1
                if window and not was_window and not retain:
                    for cache in windows:
                        cache.clear()
                was_window = window
                for layer, row in requested.items():
                    cache = windows[layer]
                    misses = row.keys() - cache.keys() if window else row.keys()
                    count = len(misses)
                    target = ledger[phase]
                    target["expertMisses"] += count
                    target["bytes"] += sum(experts_by_layer[layer][expert] for expert in misses)
                    # The Ornith 35B Q4_K_M layout has three separate families.
                    target["logicalReadSpans"] += 3 * (span_count(misses) if retain else count)
                    if window:
                        for expert in row:
                            cache[expert] = None
                            cache.move_to_end(expert)
                        while len(cache) > width:
                            cache.popitem(last=False)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--physical", type=int, choices=(1, 4, 8, 16), default=4)
    parser.add_argument("--slots", type=int, choices=[0, *range(8, 257)], default=25)
    args = parser.parse_args()
    with args.trace.open() as source:
        meta = json.loads(next(source))
        events = [json.loads(line) for line in source]
    if meta.get("kind") != "metadata" or meta.get("model_sha256") != (
            "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"):
        parser.error("expected the verified Ornith 35B Q4_K_M trace")
    print(json.dumps({
        "scope": "recorded routes; projected logical demand reads, no device inference",
        "physicalBatch": args.physical, "decodeSlotsPerLayer": args.slots, "routeEvents": len(events),
        "legacy": replay(meta, events, args.physical, args.slots, False),
        "candidate": replay(meta, events, args.physical, args.slots, True),
    }, indent=2))


if __name__ == "__main__":
    main()
