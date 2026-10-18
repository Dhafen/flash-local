# Bonsai feasibility review

Inspection date: 2026-10-18. This is source/metadata research, not a phone
benchmark or an integrated backend. No Bonsai model weights were downloaded.
The installed Ornith GGUF and its production runner remain unchanged.

## What makes the files small

Bonsai starts from pretrained Qwen weights and changes their representation.
Binary weights use two signs and one FP16 scale per 128 weights, approximately
1.125 effective bits per weight. Ternary weights have three possible values;
the deployed packing also needs group scales. Dedicated kernels consume the
packed values without materializing the entire FP16 model.
[Publisher description](https://prismml.com/news/bonsai-27b).

Bonsai 2 additionally stores weights in a fixed signed Hadamard basis. The
runtime must apply the matching activation transform and validate its
metadata. Its custom PTQ1_0/PQ2_0 packs are not interchangeable with ordinary
GGUF quantization. A generic loader, or an unrelated model packed into these
bit widths, does not establish correct inference.
[Bonsai 2 technical card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

## Candidate tradeoffs

| Candidate | Evidence and constraint |
| --- | --- |
| Original binary Bonsai 27B | Qwen3.6-27B derivative. The publisher reports about 11 token/s on iPhone 17 Pro Max with MLX Swift. This is their measurement. Its aggregate quality is lower than the original full-precision model, including tool-use scores. It is a separate model/quality decision. |
| Ternary Bonsai 2 27B | Qwen3.8-27B derivative. The publisher reports 98.2% of its aggregate full-precision benchmark score. The smallest published language GGUF is 5,946,648,928 bytes, already larger than our 5.5 GB process ceiling before runtime state. No comparable verified iPhone throughput was found in the inspected material. |

Sources: [binary MLX card](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit),
[Bonsai 2 card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
Benchmark averages are not guarantees for coding, agents, or individual tasks.
The older binary model's speed does not establish the newer model's speed.
Architectural context capacity does not demonstrate 65K within the phone budget.

The inspected [Bonsai 2 configuration](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/fcba37d2117a7077eac6b613b2668d14d9779edd/config.json)
has 16 full-attention layers, four KV heads and head dimension 256. Calculated
full FP16 K/V storage alone is 0.537 GB at 8K and 4.295 GB at 65,536 tokens
(`2 * layers * KV heads * head dimension * tokens * 2 bytes`). These are
geometry-based estimates, not measured allocation; weights, recurrent state
and workspaces are additional. Other KV strategies need separate validation.

## Source review and integration boundary

Read `Quantized.swift`, packed Metal `quantized.h`, symmetric/release
quantization tests, `Hadamard.swift`, `hadamard.h` and fused-input tests in
[Prism's MLX Swift fork](https://github.com/PrismML-Eng/mlx-swift/tree/2a673fa8324ca9900f37445db1f66f4400e4af15).
The source contains native low-bit kernels and fused activation/rotation
paths. These tests were read, not executed by this project. The two publisher
whitepapers were also read at
[Bonsai-demo db871ea](https://github.com/PrismML-Eng/Bonsai-demo/tree/db871ea96bffff5ce5609873c412b75d7cd5ba3c).

The native fork was inspected separately at
[`1180918`](https://github.com/PrismML-Eng/llama.cpp/tree/11809184898113482c2f510fd13da669aaba1052):
the dense and MoE Qwen35 graphs, Hadamard tied-output contract, and Metal
operator dispatch/fusion code. Its convolution/SiLU fusion is a same-weight
candidate; its GDN state-scatter fusion requires the newer rows/ring graph
contract absent from our pinned LES graph. Neither is a measured improvement
here, and changing the whole runtime would combine unrelated changes.

The inspected papers/code do not provide a reproduced, quality-preserving
Ornith compression result. Converting the installed Q4 weights to fewer bits
would change the model and needs its own reference and task evaluations.
Do not replace the app's shared MLX dependency globally: that would also
affect REAM and the separate QPack work. Any Bonsai trial needs isolation,
exact format/graph/tokenizer support, an honest memory plan, and physical
phone measurements before it becomes a default or a speed recommendation.

Current priority remains admitting a correct same-weight Ornith LES comparison
and collecting at least 128 generated tokens per comparable phone run.
