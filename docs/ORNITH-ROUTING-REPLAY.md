# Ornith expert routing replay

This offline experiment compares storage policies against the same ranked expert
IDs. It reads JSON, runs no inference, and neither changes the app nor requires
the GGUF. The current Ornith routing observer stores checksums and selection
counts, not ordered expert IDs. The retained
[phone build 121](ORNITH-PHONE-121.md) and
[build 123](ORNITH-PHONE-123.md) results therefore cannot be replayed. Their
returned-read counters are separate measurements and include OS-cache hits.

## Run the experiment

From the repository root, with Python 3.9 or later:

```sh
python3 scripts/replay_ornith_routing.py --demo sticky --svg /tmp/ornith-sticky.svg > /tmp/ornith-sticky.json
python3 scripts/replay_ornith_routing.py --demo churn --svg /tmp/ornith-churn.svg > /tmp/ornith-churn.json
python3 -m unittest discover -s scripts/tests -p test_replay_ornith_routing.py
```

To replay explicit routes, use `--trace path/to/trace.json` in place of
`--demo`. The input uses the existing layer-major
[preselected RoutingTrace geometry](../Sources/ModelAdapters/Streaming/RoutingTrace.swift).
A minimal example with a per-layer payload size is:

```json
{
  "num_experts": 4,
  "experts_per_token": 2,
  "tokens": 3,
  "layers": {"0": [[0, 1], [0, 2], [1, 2]]},
  "expert_bytes_by_layer": {"0": 1900000}
}
```

Each layer must include the same number of rows, in token order, with distinct
in-range IDs in router rank order. The first row is the start of the replay.
Use one continuous sequence and phase per input; replay separate prompts,
prefill, and decode separately and record each slice's origin outside the file.
Replay starts with empty caches and no trained predictor state. A trace with a
hidden-state seed or only a digest cannot stand in for explicit routes. Supply
the routed weight payload bytes per expert in each layer (all routed tensor
families combined), or use `--expert-bytes N` if uniform. It is an input assumption
unless derived from the original tensor descriptors; it is not necessarily the
resident allocation stride. A supplied trace is labeled “provided”; the tool
does not verify how it was captured. Reports and charts omit raw route rows,
prompt text, token IDs, paths and other input metadata.

`--slots` sets shared capacity per layer and must accommodate the whole top-k.
`--prefetch` is B's maximum new speculative reads per layer for the next token.
`--hold` sets C's number of tokens per window. Defaults are 16 slots, four
prefetches and eight held tokens for Ornith geometry (smaller traces cap
default slots at their expert count and depth at the slot count). The reported
expert cache budget is the sum of per-layer slots times per-expert payload.
It excludes KV/SSM state, dense weights, allocation padding, OS file cache,
and any other app memory.

## What the arms compare

| Arm | Policy | Semantics |
| --- | --- | --- |
| A | Exact LRU | Use the recorded ranked top-k, find the whole row's hits before installing misses and evict the least recently used unselected weights. The recorded order breaks recency ties within a row. |
| B | Exact LRU with causal prefetch and demand fallback | Learn which next experts historically followed the previous rank-one expert at each layer. Update only after both rows are known. Predict from the current row, ordering candidates by count and then by expert ID. Stage only nonresident candidates in the same bounded LRU before the next row. Load every actual miss, regardless of prediction. |
| C | Fixed-window routing counterfactual | At each layer, use the first recorded top-k of each window until the next boundary. Use A's cache policy. This holds the number of active experts fixed but may change their identities. It does not generate tokens or adapt router scores, downstream hidden states, or later route decisions. |

B models completion of every staged read before the next demand. It charges
all speculative bytes, including reads never consumed by a demand. It can
evict useful payloads, so it can increase both demand and total bytes. A/B's
route digests cover rank, layer and row sequence and must match; these digests
describe the replay's input route identity, not model output equivalence.
C's replaced selection share is the fraction of original selections absent
from the simulated row; changes are by set membership, so a rank-only
permutation is not counted as replacing an expert. A route digest can still
change on a rank-only permutation. C's numbers cannot establish answer quality,
the accuracy of a trained controller, or how a changed sequence would route
if it were actually generated.

JSON includes per-layer overlap, switch rate, expert frequency concentration,
entropy and a reuse gap histogram. Adjacent overlap is intersection size
divided by top-k; set switch rate is the share of adjacent rows with at least
one changed expert. Reuse gaps are distances between consecutive selections
of the same expert in **token positions**, not LRU stack distances. The SVG
shows demand/prefetch bytes and adjacent overlap across token positions and
layers. The model counts one payload per read, including repeat reads; it
excludes system I/O timing, contention, page faults, read coalescing and
device cache behavior. Bytes are neither measured flash transfers nor a
latency prediction.

## Reproducible synthetic controls

Both profiles use 128 token positions, 40 layers, 256 experts and top-8, with
16 slots per layer, up to four staged reads and an eight-token C window. Each
expert is assumed to have a 1,900,000-byte payload, for a **modeled expert cache
budget of 1.216 GB**. Sticky samples from a new pool of 20 experts every 16
tokens and carries six experts between the other adjacent rows; churn selects
eight independently at every position. These are generated controls, not
measurements of Ornith's actual routing distribution.

| Synthetic profile | Adjacent overlap | A total | B demand + prefetch = total | B unused prefetch | C total | C selections replaced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sticky | 74.58% | 12.126 GB | 13.467 + 6.019 = 19.486 GB | 5.413 GB | 7.262 GB | 39.09% |
| Churn | 3.13% | 73.106 GB | 73.011 + 8.123 = 81.134 GB | 7.566 GB | 9.154 GB | 84.82% |

A/B preserve all expert selections. This predictor raises total traffic in
both synthetic profiles; churn's smaller demand count alone would hide the
cost. C saves traffic at the cost of changing many selections. The next
measurement is to record bounded real top-k rows from several fixed public
workloads, separate prefill from decode, and replay matched byte budgets.
Only a paired phone benchmark can evaluate stalls, total system residency and
speed. Any routing change also needs independent logit, generation and task
evaluations before it is considered for use.

[Apple AFM 3 Core Advanced](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models)
uses a trained prompt-level selector with periodic reselection and many
shared experts. [Princeton's temporally extended MoE](https://arxiv.org/abs/2604.20156)
trains a per-layer controller for when to switch and which expert mask to
load on gpt-oss-20b. Their results motivate testing locality; neither architecture
is implemented by C, and their quality findings do not transfer to Ornith.
