# Build 247: Flint admission investigation

The iPhone report contains 556 routed token/layer rows. CPU and Metal did
not select identical expert sets. Of four checked logit positions, position 6
also had a different top token; maximum absolute error there was 6.18 and RMS
error was 2.34. The existing admission correctly rejected the model. These
numbers do not establish which implementation is correct.

The original control varies several parameters at once: CPU uses F32 KV,
Flash Attention off, 256 context and physical batch 1; the ordinary Metal
candidate uses F16 KV, Flash Attention on and the requested configuration.
The comparison now records the earliest differing route and, after failure,
attempts a separate Metal run matching the CPU context and batch with F32 KV
and Flash Attention off. This additional run is diagnostic only; the original
gate still decides whether serving may start. Existing files and model weights
are not rewritten.

Pinned ggml's CPU Q4_1 dot product quantizes activations to Q8_1. The Metal
Q4_1 matrix-vector kernel consumes F32 activations; batched matrix kernels
can stage values in F16, so GPU precision depends on the selected kernel. A
small local experiment read 414,720
bytes of actual Q4_1 weights from 27 expert-tensor slices and used synthetic
inputs. The observed median relative RMS differences were approximately 0.55%
to 0.88% across input distributions. This does not identify the source of the
phone mismatch or demonstrate end-to-end correctness. Full reference logits,
Metal execution and admission must be checked on the actual iPhone.

A second bounded probe invoked the compiled pinned GGML CPU dot product using
synthetic activations and compared it with dequantized Q4_1 times F32 inputs,
accumulated independently in FP64. It sampled 27 expert slices (414,720
bytes, median relative RMS 0.55%) and 14 slices from the two leading dense
layers (409,600 bytes, median relative RMS 0.46%). These probes did not
execute Metal or the full model and did not rehash the entire model file.

No phone performance improvement, full-model numerical correctness, or Splash
speedup is established by these changes.
