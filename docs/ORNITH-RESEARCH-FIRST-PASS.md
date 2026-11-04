# Ornith on iPhone 17 Pro Max: first research pass

## Release status and reproducibility

GitHub Actions [CI #424](https://github.com/Dhafen/flash-local/actions/runs/41858653979)
used source `a50f027388bdf7973525d6b63a31032ed2ba7242`. All eight QPACK/LES jobs
across four attempts had runner ID 0 and no started steps. GitHub annotations
say the hosted runner was not acquired. Queue time was 901–902 seconds per job;
execution time is unobserved. The app job was skipped. The preceding launcher
only dispatched CI. Build 420 is the last verified internal delivery in this
sequence and predates the benchmark fix.

```sh
gh api 'repos/Dhafen/flash-local/actions/runs/41858653979/attempts/4/jobs?per_page=100' > /tmp/ornith-ci-attempt.json
python3 scripts/summarize_ci_attempt.py /tmp/ornith-ci-attempt.json
python3 -m unittest discover -s scripts/tests -q
```

The CI change puts portable tests on Ubuntu before macOS, queues QPACK before
LES, and serializes private deliveries without cancelling an active one. GitHub
may replace an older pending run in the same concurrency group. No Xcode,
numerical, signing or upload gate is removed. These changes reduce simultaneous
runner demand and identify earlier failures; they cannot create runner capacity.

## Physical evidence and first bottleneck

The user's recorded iPhone18,2 build 414 trace used 4,096 tokens of configured
context, source label `b340f263c16f-dirty` and the pinned Q4_K_M artifact. It is a
short, single-run baseline. Historical native page-cache results used separate
protocols and cannot establish a speedup relative to this run.

| Measurement | Recorded build 414 |
| --- | ---: |
| Prefill, 90 evaluated tokens and 0 reused | 86.260 s / 1.043 tokens/s |
| Decode, 16 completed forwards | 10.871 s / 1.472 forwards/s |
| Prefill / decode returned I/O | 23.514 GB / 2.656 GB |
| Prefill / decode returned I/O per evaluated token | 261.27 MB / 166.00 MB |
| Peak observed process footprint | 3.885 GB |
| Prefill projection compute | 41.351 s |
| Prefill projection I/O wait | 14.002 s |

Returned bytes may be served by the OS page cache. They do not measure physical
NAND traffic. Profiling counters are phase aggregates and must not be summed as
a disjoint decomposition of wall time. Exact TTFT, thermal and energy results
are unavailable in this trace. Eliminating only the recorded prefill I/O wait
would still leave about 72 s if the other timings held and all of that wait were
on the critical path. The immediate experiment is paired prefill at fixed
tokens, weights and context with exact prefix reuse and physical batching,
recording TTFT, projection compute, returned reads and phase-specific rates.
A 32K build 418 DFlash journal contains timestamps and footprint samples without
a complete paired speed or quality report.

## Weights and gradient-free controls

The pinned MIT Ornith original revision is
`10fbf86fed7ecee4a061f8b499a618f46001cac1`. Its 16 original shards are
71,903,871,064 bytes including headers, beyond the approximately 60 GiB free
in this workspace. The app uses a distinct 21,713,463,040-byte Q4_K_M
conversion at revision `12393612fd4f730ff5aadc23e9b8f9648aa49ceb`,
SHA-256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
The source model has 40 decoder layers, 256 routed experts per layer, top-8
routing and one shared expert path. Full-model quality was not evaluated.

[The published Kinetix replay](../scripts/WEIGHT-SPACE-REPLAY.md) independently
checks archived weight edits, real CPU rollouts and exact rollback. On the
same published level with fresh paired action randomness the result was
0/32 baseline, 32/32 edited and 0/32 restored.

The Ornith probe downloads and verifies one **real original** BF16 router
tensor, 1,048,576 bytes, with exact bounded ranges and SHA-256. With NumPy 2.3.5:

```sh
OPENBLAS_NUM_THREADS=1 python3 scripts/probe_ornith_router.py
```

It edits one expert row on a copy, rounds back to BF16 and restores the
original bits after comparison. With synthetic hidden states, expert 206
appeared in 5/32 versus 32/32 selected fitting rows and 10/128 versus 11/128
independent holdout rows. The holdout changed seven top-8 sets. This is a narrow intervention and a
negative generalization control, not evidence of better routing locality,
quality or speed. No weight payload or patch was deployed to the app.

[Routing replay](ORNITH-ROUTING-REPLAY.md) documents exact LRU, exact
prediction with fallback, and a fixed-window routing counterfactual. Causal
prefetch increased total modeled I/O in both synthetic scenarios. The held
routing scenario traded reduced traffic for changes to 39–85% of selections.
The phone trace currently exports aggregate counts or hashes, so no real
expert-ID sequence was replayed.

## Candidate order for physical benchmarking

| Candidate | Text weights | Role |
| --- | ---: | --- |
| [Qwen3.5-4B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/blob/main/Qwen3.5-4B-Q4_K_M.gguf) | 2.74 GB | Small resident latency control |
| [Gemma 4 E4B QAT](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/main) | 5.15 GB | Dense mobile control, admission to be measured |
| [Ornith 1.5 9B Q4_K_M](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/tree/main) | 5.78 GB | Same-family quality and latency control |
| [gpt-oss-20b MXFP4](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/tree/main) | 12.1 GB | Streaming comparison, larger individual experts |
| [Gemma 4 26B-A4B QAT](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/tree/main) | 14.4 GB | MoE comparison |
| [Qwen3-30B-A3B Q4_K_M](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/blob/main/Qwen3-30B-A3B-Q4_K_M.gguf) | 18.6 GB | MoE comparison |

File size alone does not establish runtime compatibility, RAM use or speed.
Use the existing live-phone suite, `python3 scripts/flash_local.py test --help`,
with paired public prompts, explicit seeds, executable grading and separate
4K/16K/32K/65K runs where admitted. Record the source, model and harness hashes.
Compare the exact runtime route with prefetch disabled/enabled before adding
a quality-changing router. No candidate was run on an iPhone in this pass.
