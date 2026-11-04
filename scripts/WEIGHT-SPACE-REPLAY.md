# Replaying a direct edit of trained weights

This research control reconstructs the car policy in Experiment 4 of
[Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
The experiment used a trained Kinetix reinforcement-learning policy whose
car could reach a ramp but stalled there. GPT-6 Astra authored an edit
that wires a small path through existing units: the car reverses to gain
runway, drives back towards the ramp and releases the override after
crossing. The authors report 0/256 baseline successes and 1,024/1,024
edited successes on a separate action sampling stream.

The local script replays the *published* edit; it does not ask an editor
to rediscover it. It reads four SHA-256-pinned inputs from the
[authors' experiment release](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car)
(logs and networks under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
It rounds the trained PPO actor using the prompt's scales, checks every
value against the prompt and preserves the sign of rounded negative
zeros. The harness applies a restricted numeric edit syntax in order,
without executing the archived reply. It requires the resulting 14
tensors to match the archived edited actor byte for byte. It reverses
the edits in memory, in reverse order, and requires the original tensor
digest to match. No training, gradient calculation, runtime changes or
GGUF writes are involved.

## Run

From the Flash Local repository root with Python 3.12:

```sh
python3 -m venv /tmp/flash-weight-replay-env
. /tmp/flash-weight-replay-env/bin/activate
python -m pip install 'numpy==2.3.5'

git clone --filter=blob:none --no-checkout \
  https://github.com/RikiyaT/lm_understand-logs.git /tmp/flash-weight-replay-source
git -C /tmp/flash-weight-replay-source sparse-checkout set exp4_fix_car
git -C /tmp/flash-weight-replay-source checkout --detach 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f

python scripts/replay_kinetix_weight_edit.py /tmp/flash-weight-replay-source/exp4_fix_car
python -m unittest discover -s scripts/tests -p test_replay_kinetix_weight_edit.py
```

For an independent paired CPU rollout on the public Kinetix level:

```sh
python -m pip install 'kinetix-env==3.0.2' 'jax[cpu]==0.9.0' 'jax2d==1.0.1'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  python scripts/replay_kinetix_weight_edit.py /tmp/flash-weight-replay-source/exp4_fix_car \
  --episodes 32 --seed 20261104
```

This uses a fresh stream of 32 paired episode keys and the same exact
initial scene for all three actors. The action keys are independent
across episodes and shared across the baseline, edited and restored
actors. The six actions are sampled from independent categorical
distributions as specified by the source. Each episode stops at
termination or 256 steps. Evaluation keeps the batch size at eight
to bound compile size and memory. The report includes source hashes,
edit and rollback checks, seed, package versions, outcome counts and
first motor actions.

## Observed result

On a Linux x86-64 CPU with seed `20261104`, the 51 edit statements
changed 6,073 of 171,664 entries (3.5377%). The edited tensors matched
the authors' archive byte for byte. Reversing the edits restored the
exact baseline tensor digest and matched its recorded results on every
paired episode.

| Actor | Successes | Timeouts | Mean steps | First motor action |
|---|---:|---:|---:|---|
| Baseline | 0 / 32 | 32 | 256 | Off, 32 / 32 |
| Edited | 32 / 32 | 0 | 73 | Reverse, 32 / 32 |
| Restored | 0 / 32 | 32 | 256 | Off, 32 / 32 |

There were no red contacts or invalid episodes in any arm.

This is an independent rerun on the same task, with new action randomness.
It is not the authors' exact evaluator or a holdout over unseen levels.
It also establishes no effect on Ornith, mobile throughput or
instruction-following quality. Before considering a language-model
weight edit for Flash Local, use a separate small model, a frozen
artifact and decoding budget, and preregistered paired target and
capability-regression sets. Establish any benefit on unquantized weights
before testing whether it survives conversion to a separate GGUF.
