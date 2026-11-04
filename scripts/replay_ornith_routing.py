#!/usr/bin/env python3
"""Offline replay of expert IDs; reads no model weights and runs no inference."""
import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys


@dataclass(frozen=True)
class Trace:
    num_experts: int
    top_k: int
    tokens: int
    layers: dict
    bytes_by_layer: dict


def integer(value, label, lower, upper=2**64 - 1):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"invalid {label}")
    return value


def layer_number(key):
    if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or len(key) > 3:
        raise ValueError("invalid layer key")
    layer = integer(int(key), "layer number", 0, 255)
    if key != str(layer):
        raise ValueError("layer keys must use canonical decimal")
    return layer


def read_trace(document, uniform_bytes):
    if not isinstance(document, dict):
        raise ValueError("expected a routing trace object")
    raw_layers = document.get("layers")
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise ValueError("replay requires explicit expert rows; routing digests and seeds cannot be replayed")
    experts = integer(document.get("num_experts"), "num_experts", 1, 256)
    top_k = integer(document.get("experts_per_token"), "experts_per_token", 1, experts)
    tokens = integer(document.get("tokens"), "tokens", 1, 65536)
    layers = {}
    for key, rows in raw_layers.items():
        layer = layer_number(key)
        if not isinstance(rows, list) or len(rows) != tokens:
            raise ValueError("every layer must have the declared number of token rows")
        parsed = []
        for row in rows:
            if not isinstance(row, list) or len(row) != top_k:
                raise ValueError("every row must contain top-k expert IDs")
            ids = tuple(integer(expert, "expert ID", 0, experts - 1) for expert in row)
            if len(set(ids)) != top_k:
                raise ValueError("expert IDs in a row must be distinct")
            parsed.append(ids)
        layers[layer] = parsed
    byte_map = document.get("expert_bytes_by_layer")
    if uniform_bytes is not None:
        if byte_map is not None:
            raise ValueError("choose either --expert-bytes or expert_bytes_by_layer")
        size = integer(uniform_bytes, "expert byte size", 1)
        byte_map = {layer: size for layer in layers}
    else:
        if not isinstance(byte_map, dict):
            raise ValueError("an expert payload size is required for byte accounting")
        byte_map = {layer_number(key): integer(value, "expert byte size", 1)
                    for key, value in byte_map.items()}
        if byte_map.keys() != layers.keys():
            raise ValueError("expert_bytes_by_layer must match layers")
    return Trace(experts, top_k, tokens, dict(sorted(layers.items())), byte_map)


def synthetic_trace(profile, uniform_bytes):
    # Only the public RNG random() stream is used so sampling is reproducible.
    rng = random.Random(35258)

    def draw(pool, count):
        pool = list(pool)
        return tuple(pool.pop(int(rng.random() * len(pool))) for _ in range(count))

    layers = {}
    for layer in range(40):
        previous = ()
        for token in range(128):
            if profile == "churn":
                selected = draw(range(256), 8)
            else:
                if token % 16 == 0:
                    pool = draw(range(256), 20)
                    selected = draw(pool, 8)
                else:
                    carry = draw(previous, 6)
                    selected = carry + draw((expert for expert in pool if expert not in carry), 2)
            layers.setdefault(layer, []).append(selected)
            previous = selected
    # Illustrative payload, not a measured Ornith expert extent.
    size = 1_900_000 if uniform_bytes is None else integer(uniform_bytes, "expert byte size", 1)
    return Trace(256, 8, 128, layers, dict.fromkeys(layers, size))


def summarize_layer(rows):
    previous = set()
    frequencies, gaps, last_seen = Counter(), Counter(), {}
    overlap = []
    for position, row in enumerate(rows):
        selected = set(row)
        if position:
            overlap.append(len(previous & selected) / len(row))
        for expert in row:
            frequencies[expert] += 1
            if expert in last_seen:
                gaps[position - last_seen[expert]] += 1
            last_seen[expert] = position
        previous = selected
    selections = sum(frequencies.values())
    distribution = [count / selections for count in sorted(frequencies.values(), reverse=True)]
    return dict(distinct_experts=len(frequencies),
                top_expert_share=distribution[0],
                entropy_bits=max(0., -sum(p * math.log2(p) for p in distribution)),
                adjacent_overlap=sum(overlap) / len(overlap) if overlap else None,
                adjacent_set_switch_rate=sum(value < 1 for value in overlap) / len(overlap) if overlap else None,
                reuse_token_gap_counts=dict(sorted(gaps.items()))), overlap


def replay_layer(rows, slots, prefetch):
    # Value is True until a demand uses that prefetched payload. Both demand
    # and prefetch use the same slots, so speculation can evict useful weights.
    cache = OrderedDict()
    learned = defaultdict(Counter)
    counts = Counter()
    previous_lead = None
    for position, row in enumerate(rows):
        selected = set(row)
        for expert in row:
            if expert in cache:
                counts["demand_hits"] += 1
                counts["useful_prefetch_reads"] += cache[expert]
                cache[expert] = False
            else:
                counts["demand_reads"] += 1
        # Preserve all current-row hits before installing misses. Expert IDs
        # within a row are not a series of independent demand transactions.
        missing = sum(expert not in cache for expert in row)
        evictions = max(0, len(cache) + missing - slots)
        for expert in list(cache):
            if evictions == 0:
                break
            if expert not in selected:
                del cache[expert]
                evictions -= 1
        for expert in row:
            cache[expert] = False
            cache.move_to_end(expert)

        if prefetch:
            lead = row[0]
            if previous_lead is not None:
                learned[previous_lead].update(row)
            previous_lead = lead
            if position + 1 < len(rows):
                candidates = sorted(learned[lead], key=lambda expert: (-learned[lead][expert], expert))
                predicted = [expert for expert in candidates if expert not in cache][:prefetch]
                for expert in reversed(predicted):
                    if len(cache) == slots:
                        cache.popitem(last=False)
                    cache[expert] = True
                    counts["prefetch_reads"] += 1
    return counts


STRATEGIES = (
    ("A", "Exact LRU"),
    ("B", "Exact with prefetch"),
    ("C", "Fixed-window counterfactual"),
)


def replay(trace, slots, prefetch, hold, source, demo=None):
    integer(slots, "slots (must fit the whole top-k)", trace.top_k, trace.num_experts)
    integer(prefetch, "prefetch depth", 0, slots)
    integer(hold, "hold length", 1, 65536)
    totals = {arm: Counter() for arm, _ in STRATEGIES}
    hashes = {arm: hashlib.sha256() for arm, _ in STRATEGIES}
    layers, overlaps = [], []
    selections = trace.tokens * len(trace.layers) * trace.top_k
    for layer, original in trace.layers.items():
        stats, overlap = summarize_layer(original)
        overlaps.append(overlap)
        strategies = {}
        size = trace.bytes_by_layer[layer]
        fixed = [original[position - position % hold] for position in range(trace.tokens)]
        for arm, _ in STRATEGIES:
            routes = fixed if arm == "C" else original
            hashes[arm].update(json.dumps((layer, routes), separators=(",", ":")).encode("ascii"))
            raw = replay_layer(routes, slots, prefetch if arm == "B" else 0)
            shared = [len(set(current).intersection(native)) for current, native in zip(routes, original)]
            data = dict(
                demand_hits=raw["demand_hits"],
                demand_reads=raw["demand_reads"],
                prefetch_reads=raw["prefetch_reads"],
                useful_prefetch_reads=raw["useful_prefetch_reads"],
                unused_prefetch_reads=raw["prefetch_reads"] - raw["useful_prefetch_reads"],
                demand_read_bytes=raw["demand_reads"] * size,
                prefetch_read_bytes=raw["prefetch_reads"] * size,
                total_read_bytes=(raw["demand_reads"] + raw["prefetch_reads"]) * size,
                useful_prefetch_bytes=raw["useful_prefetch_reads"] * size,
                unused_prefetch_bytes=(raw["prefetch_reads"] - raw["useful_prefetch_reads"]) * size,
                changed_route_rows=sum(count < trace.top_k for count in shared),
                replaced_expert_selections=sum(trace.top_k - count for count in shared),
            )
            totals[arm].update(data)
            strategies[arm] = data
        layers.append(dict(layer=layer, expert_payload_bytes=size, **stats, strategies=strategies))
    return dict(
        schema="ornith-routing-replay-v1", trace_source=source, synthetic_profile=demo,
        trace_tokens=trace.tokens, num_experts=trace.num_experts, top_k=trace.top_k,
        route_rows=trace.tokens * len(trace.layers), selections=selections,
        slots_per_layer=slots, prefetch_depth=prefetch, hold_tokens=hold,
        expert_cache_budget_bytes=slots * sum(trace.bytes_by_layer.values()),
        quality_evaluated=False, physical_io_measured=False, latency_measured=False,
        strategies=[
            dict(arm=arm, label=label, **totals[arm], route_digest=hashes[arm].hexdigest(),
                 replaced_selection_share=totals[arm]["replaced_expert_selections"] / selections,
                 demand_hit_rate=totals[arm]["demand_hits"] / selections)
            for arm, label in STRATEGIES
        ], layers=layers,
    ), overlaps


def render_svg(report, overlaps):
    width, chart_left, chart_width = 960, 280, 540
    row_height = max(5, min(20, 380 // len(overlaps)))
    grid_top = 335
    grid_bottom = grid_top + row_height * len(overlaps)
    height = grid_bottom + 94
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Expert routing replay</title>',
        '<desc id="desc">Modeled read traffic by strategy and consecutive-token overlap by layer.</desc>',
        '<style>text{font-family:system-ui,sans-serif;fill:#172b41}.muted{fill:#496071}.small{font-size:13px}'
        '.section{font-size:17px;font-weight:700}.title{font-size:28px;font-weight:700}</style>',
        f'<rect width="{width}" height="{height}" rx="14" fill="#f5f8fb"/>',
    ]

    def label(x, y, value, kind="small", extra=""):
        lines.append(f'<text x="{x}" y="{y}" class="{kind}" {extra}>{value}</text>')

    label(34, 47, "Expert routing replay", "title")
    source = (f"Synthetic {report['synthetic_profile']} trace" if report["trace_source"] == "synthetic"
              else "User-provided expert trace")
    label(34, 73, f"{source} · {report['trace_tokens']} tokens · {len(overlaps)} layers · "
                  f"top-{report['top_k']} · {report['slots_per_layer']} slots/layer", "small muted")
    label(34, 113, "Modeled expert payload reads", "section")
    lines.extend([
        '<rect x="280" y="130" width="12" height="12" fill="#285f98"/>',
        '<rect x="408" y="130" width="12" height="12" fill="#bb6418"/>',
    ])
    label(299, 141, "Demand reads")
    label(427, 141, "Prefetch reads")
    peak = max(1, *(row["total_read_bytes"] for row in report["strategies"]))
    for index, arm in enumerate(report["strategies"]):
        y = 157 + 35 * index
        label(34, y + 15, f"{arm['arm']} · {arm['label']}")
        demand = arm["demand_read_bytes"] / peak * chart_width
        prefetch = arm["prefetch_read_bytes"] / peak * chart_width
        lines.append(f'<rect x="{chart_left}" y="{y}" width="{chart_width}" height="20" rx="2" fill="#e3eaf1"/>')
        lines.append(f'<rect x="{chart_left}" y="{y}" width="{demand:.3f}" height="20" fill="#285f98"/>')
        lines.append(f'<rect x="{chart_left + demand:.3f}" y="{y}" width="{prefetch:.3f}" height="20" fill="#bb6418"/>')
        byte_value = arm["total_read_bytes"]
        label(830, y + 15, f"{byte_value / 10**9:.2f} GB" if byte_value >= 10**9
              else f"{byte_value / 10**6:.2f} MB" if byte_value >= 10**6 else f"{byte_value:,} B")
    changed = report["strategies"][2]["replaced_selection_share"]
    label(34, 278, f"C replaces {changed:.1%} of selections. Replay measures neither model quality nor device I/O or latency.",
          "small muted")
    label(34, 315, "Adjacent top-k overlap by layer", "section")
    grid_left, grid_width = 96, 786
    transitions = len(overlaps[0])
    bins = min(transitions, 96)
    if not bins:
        label(grid_left, grid_top + 18, "At least two token rows are required for an overlap measurement.")
    else:
        for i, (layer, values) in enumerate(zip(report["layers"], overlaps)):
            y = grid_top + i * row_height
            if i in (0, len(overlaps) - 1) or (row_height >= 9 and i % 5 == 0):
                label(34, y + max(5, row_height // 2), f"L{layer['layer']}",
                      "small muted", 'dominant-baseline="middle"')
            for j in range(bins):
                start, end = j * transitions // bins, (j + 1) * transitions // bins
                mean = sum(values[start:end]) / (end - start)
                rgb = tuple(round(low + (high - low) * mean)
                            for low, high in zip((230, 237, 246), (13, 82, 133)))
                color = "#{:02x}{:02x}{:02x}".format(*rgb)
                x = grid_left + j * grid_width / bins
                cell = grid_width / bins
                lines.append(f'<rect x="{x:.3f}" y="{y}" width="{cell:.3f}" height="{row_height}" fill="{color}">'
                             f'<title>Layer {layer["layer"]}, positions {start + 1}–{end}: {mean:.1%}</title></rect>')
        label(grid_left, grid_bottom + 21, "1", "small muted")
        label(grid_left + grid_width, grid_bottom + 21, transitions, "small muted", 'text-anchor="end"')
        label(376, grid_bottom + 21, "Token position (zero-based; compared with previous token)", "small muted")
        label(34, grid_bottom + 48, "Light = 0% overlap  ·  Dark = 100% overlap"
              + ("  ·  Cells average consecutive positions" if bins < transitions else ""), "small muted")
    return "\n".join(lines + ["</svg>", ""])


def unique_keys(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError("duplicate JSON key")
        values[key] = value
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--trace", type=Path, help="layer-major JSON containing explicit expert IDs")
    inputs.add_argument("--demo", choices=("sticky", "churn"), help="synthetic 40-layer/top-8 trace")
    parser.add_argument("--slots", type=int, help="per-layer shared capacity (default: min(top-k + 8, experts))")
    parser.add_argument("--prefetch", type=int, help="max new experts staged per next row (default: up to 4)")
    parser.add_argument("--hold", type=int, default=8, help="C's fixed window in tokens (default: 8)")
    parser.add_argument("--expert-bytes", type=int, help="uniform payload bytes, or set expert_bytes_by_layer in JSON")
    parser.add_argument("--svg", type=Path, help="optional chart output")
    args = parser.parse_args(argv)
    try:
        if args.demo:
            trace = synthetic_trace(args.demo, args.expert_bytes)
            source = "synthetic"
        else:
            document = json.loads(args.trace.read_bytes(), object_pairs_hook=unique_keys)
            trace = read_trace(document, args.expert_bytes)
            source = "provided"
        slots = args.slots if args.slots is not None else min(trace.top_k + 8, trace.num_experts)
        prefetch = args.prefetch if args.prefetch is not None else min(4, slots)
        report, overlaps = replay(trace, slots, prefetch, args.hold, source, args.demo)
        if args.svg:
            if args.trace and (args.svg.resolve() == args.trace.resolve() or
                               args.svg.exists() and args.svg.samefile(args.trace)):
                raise ValueError("chart output must be different from trace input")
            args.svg.write_text(render_svg(report, overlaps), encoding="utf-8")
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        print("error: could not read routing JSON or write chart", file=sys.stderr)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
