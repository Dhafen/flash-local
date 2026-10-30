# Flint build 250 numerical investigation

The user supplied a report from an iPhone running the existing official
Q4_1/F32 GGUF. The original CPU/Metal check first chose different top-six
experts at token position 0, layer 6, and differed in 128 of 556 recorded
token/layer rows. A second Metal control using F32 KV storage, Flash Attention
off and physical batch 1 still differed first at the same point (130 of 556
rows). In both comparisons, the top logit at position 6 also differed. This
does not establish that either full model result is correct.

The ordinary GGML CPU Q4_1 dot product quantizes F32 input activations to
Q8_1; the Metal vector kernel instead consumes F32 activations. A CPU/Metal
comparison therefore also contains an activation-precision difference. The
diagnostic reference added here reads the same Q4_1 bytes, uses F32 input
activations and accumulates in double, without modifying the model, normal
inference kernel or expert-routing algorithm.

`scripts/verify_flint_controls.py` validates this reference against an
independent NumPy/FP64 full graph for three untrained Q4_1 fixture models,
including 256-expert routing, resident and streamed expert weights, physical
batches 1, 4 and 8, 40-token traces, and a 300-token sliding-window case.
All 13 reference cases on the development host had identical expert sets and
top tokens; the worst absolute logit error was below 0.000003. Native API
controls also check that the diagnostic switch is CPU-only and that unloading
restores the ordinary CPU result exactly. These are synthetic CPU tests, not
proof of iPhone or trained-model correctness.

When the prior CPU/Metal gate fails, the app now also compares the original
Metal candidate to this independently tested CPU reference. Admission still
requires complete identical top-six expert sets, all four full-logit checks
within the existing limits, and an additional comparison of the callback-free
serving graph. If any control fails or cannot complete, admission fails closed
and the report retains the original and alternative comparisons. The reference
is only used during this diagnostic, not while serving chat or API requests.

The cause of the iPhone mismatch, actual trained-model numerical agreement,
memory use, latency and performance still require a new physical-iPhone run.
The short check does not verify full model quality or long-context behavior.
