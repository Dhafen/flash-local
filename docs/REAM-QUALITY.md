# Model quality and the Qwen3.8-27B target

The user's minimum is Qwen3.8-27B-class coding, reasoning and agent capability.
A faster small model alone does not satisfy that target. No trained-model
quality comparison has been completed by this project.

The user has since selected Ornith 1.5 35B-A3B as the primary candidate.
See [the pinned artifact and implementation status](ORNITH-IMPLEMENTATION.md).
Its official Q4_K_M is 21.7 GB and retains the complete main trunk rather
than the installed REAM artifact's expert/PLE pruning. That does not itself
establish 27B-class quality or an iPhone speed improvement. Publisher agent
scores use thinking and much longer contexts than the initial phone policy.

## Current artifact and alternatives

The [original REAM release](https://huggingface.co/Akicou/Qwen3.8-Flash-Next-REAM-60Pct/blob/7c9b3f5eddac3de7975aa279d00b0859283b76d8/README.md)
merges 512 experts to 308 and explicitly describes itself as experimental and
not benchmarked. The [installed GGUF's publisher](https://huggingface.co/RemySkye/Qwen3.8-Flash-Next-REAM-60Pct-GGUF/blob/28a55fecdca51a498900aeb34a4e7e8338977cd3/README.md)
also zeroed 12 of 16 logical PLE heads without retraining or calibration. Its
56.1 GB size is not evidence of retained quality. These changes are distinct
from this app's execution optimizations, which retain the file's bytes and
test logits against the same reference semantics.

[GSQ-RCO Qwen3.8-Flash-Next](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09)
is a candidate worth testing: its two Q2_0 shards total 66,423,878,624 bytes,
retain the original expert/head counts, and the authors publish quality
measurements. This does not establish parity with Qwen3.8-27B under our task
conditions. The allocation manifest contains thirteen formats despite the
Q2_0 filename. Five are unsupported by this app: Q2_0, Q3_K, Q5_K, IQ4_NL and
IQ4_XS. Exact bounded reads of both shard headers confirm `qwen4exp`, 512
experts/top-10, sixteen distinct PLE heads and compression ratio 4 at every
fourth layer. The current decoder explicitly refuses this QSA attention path.
Supporting it requires that implementation, shard-aware verified storage,
the five formats and validated tokenizer/template behavior. No weight payload
was fetched. It cannot currently replace the installed model in the app.

The two headers total 11,024,864 bytes (11,024,672 plus 192). Their SHA-256s
are `bf118cd5acbabeae0459b52a6b8cb94cc00acbf89b39fb72b337975331abefb9`
and `60a762da9127f76c0a895de009e7245566708ee638c2ae57f0c12b20ae948e4d`.
All 1,224 tensor names are unique across the shards and their calculated
packed extents fit without overlap. Expert bytes for ten selected experts
in each layer total 663,552,000 per token before reuse; the installed REAM
artifact's corresponding total is 1,425,408,000. These are weight-layout
arithmetic, not actual reader traffic, physical flash speed, memory usage or
quality measurements. QSA and dense/PLE work still have to be executed.

The [official baseline](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
uses a different dense architecture. Its source revision is pinned, but an
exact evaluated quantized artifact and local run are still required. Published
scores for the original Flash model do not transfer automatically to REAM,
PLE pruning, another quantization or this runtime.

## Paired local evaluation contract

`scripts/compare_model_quality.py` compares reports from already completed
local runs. It is an aggregation/validation tool, not a benchmark runner or
proof that the reported executions occurred. It fetches no models, executes
no generated code and uploads nothing. There are currently no real reports
to compare. Its unit-test reports are invented fixtures, not model results.

Run both models against the same versioned prompts, deterministic local
tools and grading harness. Grade coding with executable tests, reasoning
with checked answers, agent work with recorded tool outcomes, and security
review with known defects in isolated local fixtures. Use identical thinking,
sampling, context and output limits. Keep unfinished, timed-out and ungraded
tasks in the report as failures. This app currently uses greedy non-thinking
text chat; structured tool calling and the eventual LAN service need their
own implementation and compatibility tests before they can be evaluated.

Each input JSON has these schema-1 fields:

| Field | Required evidence |
| --- | --- |
| `schemaVersion` | Integer `1` |
| `inferenceLocation` | `local_iPhone` or `local_PC` |
| `model` | `upstreamRepository`, 40-hex `upstreamRevision`, 64-hex `artifactSHA256`, explicit `quantization` |
| `runtimeSourceRevision` | 40-hex commit of the evaluated runtime |
| `datasetSHA256`, `graderSHA256` | 64-hex hashes of the exact task set and grading harness |
| `generationPolicy` | `thinking`, `contextTokens`, `maximumNewTokens`, `sampling` |
| `tasks` | Every task's `id`, `category`, `promptSHA256`, Boolean `passed`, integer `checksExecuted`, `finishReason` |

Thinking is `off`, `low`, `medium` or `xhigh`. Sampling is either
`{"kind":"greedy","seed":0}` or an explicit categorical object with exactly
`kind`, `seed`, `temperature`, `topP` and `topK`. The seed is a 32-bit unsigned
integer. Greedy seeds identify the declared policy even though sampling is
deterministic. All four task categories (`coding`, `reasoning`, `agent`,
`security_review`) must be present. Only a tested output with finish reason
`stop` can pass; `length`, `cancelled` and `error` remain failures. Baseline
and candidate must have identical coverage, prompt hashes and policy.

```sh
python3 scripts/compare_model_quality.py \
  --baseline baseline-local.json --candidate candidate-local.json \
  --output paired-quality.json
```

Exit 0 means the candidate matched or exceeded the baseline's observed pass
count in every category of those supplied tasks; exit 1 means a category
regressed; exit 2 refuses invalid/incomparable reports. Individual regressions
are listed even if another task offsets the category count. The aggregate
retains declared model/runtime provenance, never copies task responses, and
always marks general quality equivalence, report authenticity and speed
comparison as unestablished. Four tests, or even a clean aggregate, cannot
establish broad 27B-class quality. Keep actual timing, memory, tool traces
and grading evidence locally beside the original reports for review.
