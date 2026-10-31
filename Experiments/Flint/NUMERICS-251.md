# Flint build 251 numerical investigation

The attached iPhone report is an observation, not a successful admission.
For a 12-token control prompt it contains 556 actual top-six expert sets:
46 routed layers for each token plus the final routed layer at each of the
four requested logit positions. Full logit vectors are compared at positions
0, 3, 6 and 11. The report stores no raw routing scores, so it does not establish
whether a near-tie, accumulation precision, a kernel, or another difference
caused the first divergent selection.

| Comparison | Different expert sets | First difference | Position 6 |
| --- | ---: | --- | --- |
| Normal CPU vs normal Metal | 128/556 | token 0, layer 6 | top-1 differs; max absolute error 6.180698 |
| Normal CPU vs matched Metal (F32 KV, no Flash Attention, physical batch 1) | 130/556 | token 0, layer 6 | top-1 differs; max absolute error 8.367809 |
| CPU Q4_1/F32 reference vs normal Metal | 2/556 | token 6, layer 35 | top-1 equal; max absolute error 3.215679 |

The user additionally reported that a matched Metal Q4/F32 control had zero
route mismatches and maximum error 0.0000515. That control is not present in
the attached build 251 JSON and its exact configuration has not been
independently established here. No serving admission relies on that claim.

The normal GGML CPU Q4_1 dot product converts activations to Q8_1; the CPU
Q4_1/F32 diagnostic instead uses the original quantized weights and F32
activations with double accumulation. This distinction is real, but is not a
confirmed root cause for the remaining two route differences. Metal selects
different matrix kernels for some batch sizes. The model graph selects top-six
before streaming the selected expert weights and remapping them to slots;
the trace reads the real expert IDs before slot remapping. Route comparisons
canonicalize the set, while full logits independently check the graph outputs.

The experimental candidates leave model weights, top-k, routing, thresholds,
and the existing memory ceiling unchanged. First they try physical batch 1;
then physical batch 1 with F32 KV and Flash Attention disabled. A candidate is
eligible only if all actual route sets and four complete logit vectors pass
against the CPU Q4_1/F32 reference, and the callback-free serving graph passes
both against the traced graph and directly against that CPU reference. Failure,
missing data, cancellation or memory admission keeps chat/API unavailable.
Successful admission records its mode and binds the sealed result to the model,
native binary, executable, operating system and requested configuration. The
interface and saved generation report identify the mode actually loaded.

Only a diagnostic sample is checked; successful admission does not prove full
model quality, tool calling, long-context behavior, sustained speed, or iPhone
stability. The conservative F32-KV candidate may consume more memory and make
prefill slower. For example, with a 400 MB app footprint, a 1 GB scratch reserve
and physical batch 1, the current native formula estimates about 4.32 GB at 8K,
5.73 GB at 32K and 7.61 GB at 65K; these are estimates rather than observed
peaks. A 5.5 GB ceiling would reject the latter two under these assumptions.

The previously installed 25,992,966,976-byte Q4_1/F32 GGUF remains the input.
No model conversion, second full model copy, or replacement download is part
of this change. The verified descriptor and existing checksum checks remain.
The new native revision and admission policy require a fresh numerical check;
this does not require deleting the model or its existing storage.
