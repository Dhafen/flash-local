# Ornith LES experiment

This isolated experiment reuses the installed Ornith Q4_K_M: 21,713,463,040
bytes, SHA-256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
Production GGUF/REAM and Swiftlet/QPack paths are unchanged. Native controls
must pass before app integration; no phone throughput is claimed.

## Source inspection

Reviewed source snapshots:

- OpenBMB/MiniCPM-V-Apps `fa9d837e3d38dfde7a9484ec1cc6bc2e5bc9b420`:
  `MiniCPMModelConst.swift`, `MBMtmd.mm`, build scripts and submodule pin.
- tc-mb/llama.cpp-omni **ios-omni** `96bbf61710d44421ea54e63f5194f9b17c20d877`:
  `src/models/qwen35moe.cpp`, `llama-moe-slot.cpp`, `llama-moe-experts.cpp`,
  Metal implementation, `tools/moe/experts.py` and `strip_experts.py`.
- Anemll/Flash-iOS `4f1a69777629cda07b4fdd7d93511ed43f62fad5`:
  `FlashMoEEngine.m`, `metal_infer/infer.m`, shaders and I/O experiments.
  This snapshot has no license grant file; its code is not copied or shipped.
- Swiftlet's QPack/Engine/Metal sources upstream `de1a942241f65ac43290cecffc85d69ea5b52310` and our separately tested grouped
  dispatch fork, described in [Ornith Performance](ORNITH-PERFORMANCE.md).

The [publisher's 13+ token/s iPhone claim](https://github.com/OpenBMB/MiniCPM-V)
does not supply an exact reproducible prompt/run/thermal/configuration record.
The [LES ablation table](https://github.com/tc-mb/llama.cpp-omni/blob/96bbf61710d44421ea54e63f5194f9b17c20d877/docs/les/lookahead-expert-scheduling.md)
is explicitly for **iPad M4 16 GB**: 2.97 baseline, 9.81 scheduling/layout,
14.92 with mixed quantization, 17.39 with DSpark. These are publisher results,
not Ornith measurements. The current app defaults to Q3_K experts, top-8,
4096 context, lookahead depth 8, mmap dense weights and DSpark **off**.
Its 12 GB device memory ceiling is 6 GiB, exceeding our 5.5 decimal GB limit.
The 3 GB marketing figure is not evidence of total process peak footprint.

## Compatibility and retained mathematics

Both trunks use 40 main layers, 256 experts/top-8, 30 gated DeltaNet layers
and ten full-attention layers. The loader reads dimensions, RoPE sections,
frequency, RMS epsilon, KV heads and SSM geometry from the GGUF. Ornith also
stores an unused MTP block; main execution excludes it. A local patch avoids
predicting that unused block's experts. No routing override is enabled.
Tokenization/chat formatting continues to use the existing canonical bridge
and the original file's exact metadata. MiniCPM's vision tower is not loaded.

`experts.py` copies quantized family slices into contiguous expert-major
layer files; `strip_experts.py` retains metadata and raw dense tensor bytes.
Neither inherently requires requantization. Repacking all experts would
duplicate roughly the full installed model while preserving rollback.

Our first variant therefore copies/readback-verifies only the dense tensors
with 4 MiB buffers, preserving all raw GGUF metadata and tensor types. It
keeps expert tensors in the verified original descriptor. The local
`gguf_tensor_major` extension reads each gate/up/down span directly into its
GPU-visible destination. This costs up to three source calls per expert
instead of a contiguous preadv; device comparison must decide if repacking
is worth its storage cost. No full-model Metal mapping is created.

LES predicts the next layer from its router and the current normalized
activation. Predictions guide asynchronous prefetch; actual top-k still
determines computation, including all missed predictions. Slots remain
protected from reuse until the corresponding graph operation permits it.
The runtime uses per-layer decode windows plus shared prefill banks grouped
by tensor family and quantization stride, including mixed Q4_K/Q6_K down
projections. All buffers, prefetch slots and temporary prefill banks count
toward admission. No static MiniCPM expert-frequency/prewarm table is reused.

## Candidates and acceptance

| Candidate | Current evidence | Next decision |
| --- | --- | --- |
| A: existing packed GGUF/MLX | Phone build 107: 1.931 decode forwards/s, 190 forwards; 3.954 GB peak; uncontrolled prompt/cache | Same fixed prompt, at least 128 decode steps per mode |
| B: llama.cpp-omni LES | Build 120 serial logits satisfy the independent reference limits; ranked route checksums differ and timing remains blocked | Verify exact expert sets separately from rank permutations; keep failing batch-four disabled |
| C: Flash-MoE pipeline | Code inspected; fused expert/pointwise kernels, deferred command buffers, parallel pread; no redistribution license found | Do not copy; compare permitted independently implemented techniques |
| Swiftlet/QPack | Build 97 delivered, grouped kernels conditionally activated after four-position replay; no phone result | Different checkpoint quantization, so cannot be an identical-weight A/B |

The dynamic LES framework exports only `flo_*`; its llama/GGML symbols must
not replace the production tokenizer/reference runtime. CI verifies that
isolation for macOS, iPhone and Simulator. Aggregate instrumentation counts
all expert reads including prefetch, actual Metal dispatch calls, command
buffers and GPU command-buffer timestamp durations. GPU duration is not
wall latency; overlapping durations can sum beyond elapsed time. Application
read bytes are not physical flash-controller bytes.

Initial tests use tiny untrained fixtures only: resident versus streamed
versus lookahead CPU execution, full logits, actual router decisions, greedy
choice and recurrent/KV reset. Trained Ornith parity, 128-token sustained
throughput, OS-cache effects, memory pressure, thermals and useful 8K/32K/65K
context remain device gates. An empty application expert cache is not a
guaranteed cold OS page cache. No winning backend is selected before those
measurements. The first speed milestone is 3 token/s, then 5 or 8 if proven.

The existing backend requests `F_NOCACHE=1` on a duplicate of the verified
descriptor. Native trials explicitly apply `F_NOCACHE=0`, including when the
descriptor inherits that flag from an earlier reader. Failure to apply the
policy stops the native trial. Native cache and LES modes use both the OS
page cache and their own windows; the OS-only mode drops the decode windows.
This first comparison does not isolate a native own-cache-only policy. All
reported read bytes are explicit source reads; dense mmap page faults are
not included. Actual physical flash traffic is not exposed by these counters.

DSpark needs a matching trained draft, target verification and recurrent
rollback. MiniCPM's draft is not an Ornith draft. Speculative decoding is
disabled; no quality/distribution equivalence is claimed. The foreground
LAN API remains a follow-up after correctness and decode speed.


The iPhone streamed path requires Metal's alloc-only buffer type. Host controls
use a bounded CPU allocation adapter, while production continues to require
Metal. A separate check compares the unchanged original pinned CPU reference.
The 256-expert/top-8 fixture repeats 128 teacher-forced
positions to exercise eviction and prefetch races without trained weights.

The opt-in app benchmark uses the existing verified file capability, prepares
only dense weights, performs a bounded CPU-reference comparison, then compares
four modes over two genuine greedy generations each. Early EOS is retained
and marks that run shorter than the required 128 tokens. Batched prefill has
its own reference gate. Numerical disagreement prevents candidate benchmarks;
the original chat remains available. The default production package does not
link LES unless `FLASH_LOCAL_ENABLE_LES=1`. The dedicated delivery input requires
the native/reference/Swift gates and transfers an exact-source/hash-checked,
symbol-isolated XCFramework before app/archive/signing gates. No device result
or TestFlight delivery is established by implementing those gates.


The hosted Apple Paravirtual GPU advertises no SIMD reduction. An explicit
instruction probe also failed; it was removed rather than forcing a capability
flag. Numerical CI therefore uses a complete resident CPU oracle versus the
same streamed/cache/lookahead/page-cache graph with host allocations. That
host allocation adapter is macOS-only, requires an explicit control flag,
disables mmap for the fixture, and is restricted by the C API to at most
32 MiB. The Swift test additionally requires the generated fixture name.
It cannot enable CPU streaming on iPhone. No mathematical operator or routing
is changed. Metal is compiled for all three platforms but its runtime numerics
remain a physical-device gate. The app always uses Metal and must compare its
trained logits with the existing CPU reference before benchmarking.

## Build 107 phone evidence

The user-provided ordinary GGUF export reports 190 decode forwards in
98.381 seconds (1.931 forwards/s), 28,962,661,120 returned read bytes during
decode (152.435 MB/forward), and 3,953,939,248 bytes sampled peak footprint.
Prefill evaluates 26 new tokens while reusing 270: 17.701 seconds and
6,023,248,128 returned read bytes. Total source reads for this generation are
34,985,909,248 bytes. The screenshot shows a different, longer prefill with
506 of 511 prompt tokens completed and 90.09 GB returned reads.

The read counter is a cumulative per-generation sum of completed source
reads, including repeated extents and prefetched bytes. It is neither unique
file coverage, newly downloaded/storage bytes nor simultaneously resident
memory. Repeated expert eviction and token-at-a-time prefill can make it much
larger than the 21,713,463,040-byte file. Explicit reads can also be served by
the OS cache; these counters cannot establish physical flash traffic. The
existing chat backend keeps approximately 1.39 GB fixed weights plus 2.11 GB
evictable weights within its 3.5 GB packed-weight capacity. Further memory
holds activations, KV/SSM state, runtime and UI. Increasing history can require
many repeated reads even when the visible user message is short.

The separate LES export has zero measurements and reports `LES model loading
failed`, with a 356 MB observed footprint. Thus the above throughput belongs
to the existing engine. No LES speed, trained-model parity or fastest-backend
claim follows from these reports.

The loader investigation found another pathname operation on the verified
descriptor: manifest extent validation used `filesystem::file_size` on
`/dev/fd/N`, although expert reads already retained the descriptor directly.
The candidate fix uses `fstat` for extent validation and `F_DUPFD_CLOEXEC`
for retention, without traversing `/dev/fd`. This addresses a possible iOS
sandbox restriction; the generic build-107 report does not prove it caused
the observed failure. Bounded error-only native diagnostics now preserve the
loader's actual cause with local paths removed. Descriptor parsing/retention,
diagnostic limits/privacy and an invalid-header C-API regression join the
native controls. Physical confirmation remains required. No weights, router
selection, quantization or production GGUF/REAM computation are changed.

The host-only `ornith-les-header-control` adds the real checkpoint's full
tensor/vocabulary geometry to the small numerical fixtures. It fetches only
the pinned 10,990,848-byte metadata header, generates sparse files with no
trained weight payload, and forces `no_alloc=true`, CPU placement and no
context/decode. It checks the original loader plus three streamed slot
layouts, all 123 expert tensor types/strides/views, 41 stored blocks and the
real tokenizer IDs. Physical allocation is capped at 32 MiB per generated
file and host RSS at 1 GiB. This is descriptor/layout evidence, never trained
model numerical or GPU execution evidence. The source-bound control at
`74012d07e96aee26ea4cac3b70e8cda23fae3366` passed in Actions run
`39068453176`: 753 tensor shapes, all four loader layouts, identical tokenizer
IDs, 21,979,136 physically allocated storage bytes and 152,879,104 host peak
RSS bytes. Its report explicitly records that inference was not executed and
no trained weight payload was present. The synthetic numerical tests and
three-platform framework compilation also passed in that run.

## Adaptive prediction window follow-up

### Build 112 physical admission report and VM-buffer correction

The next phone report provides phase-boundary measurements: the first
`numerics.prediction0.prefill1` admission starts at 114,067,512 bytes after
idle cleanup and reserves 5,119,385,051 bytes in total. The following
`numerics.prediction4.prefill1` admission measures 2,216,480,384 bytes,
unchanged by allocator relief. That is 2,102,412,872 bytes above the first
baseline. It leaves only two window slots; admitting all eight real demand
slots would require 5,891,965,987 bytes, exceeding the ceiling by 391,965,987.
No numerical check or timing run completed, and the report cannot identify
whether retained allocation ownership or allocator/VM accounting caused it.

The pinned source uses `posix_memalign/free` for iOS page buffers but
`vm_allocate/vm_deallocate` for the same buffers on macOS. A recent
[upstream iOS report](https://github.com/ggml-org/llama.cpp/issues/30556)
describes elevated footprint after ownership was released, including zero
allocator relief; its proposed VM path is a plausible match, not proof of
the cause on this phone. The isolated patch enables the existing Apple VM
path symmetrically for both Metal host buffers and CPU aligned buffers.
It does not change file-backed dense mappings, tensors or mathematical ops.

Native close synchronizes queued work before destruction. Swift explicitly
closes at the end of a numerical variant or complete cold/warm pair, with
idempotent close tests retaining the old wrapper while creating new owners.
Admissions include native live-context/buffer counts, owned and mapped Metal
span bytes, and device allocation when observed. A new plan rejects any
remaining native owner. These counters are separate from physical footprint;
mapped spans are not resident RAM, and device allocation includes other app
Metal users. No live buffer or page is discarded during inference.

The added host allocator control touches nine bounded allocations, including
partial-page sizes, and checks that their VM regions disappear after free.
Host execution and three-platform compilation still cannot establish iOS
footprint recovery or trained Metal parity. The next physical acceptance
requires zero owners/buffers at every transition, recovered baseline, the
trained routing/logit gate and complete 128-token runs. The eight real
experts and 5.5 GB ceiling remain unchanged. Host gates pass and the correction
is delivered in build 113; physical confirmation remains pending.

The build-108 phone attempt stops at Swift memory admission with no completed
numerical check or benchmark; observed process peak is 2,279,560,368 bytes.
The report does not include the precise failing phase or current headroom.
It therefore does not confirm either trained-model loading or an LES speed.

The candidate releases unused MLX and Darwin allocator pages at transitions
between inference owners, after the preceding context and autorelease pool
have ended. It resamples actual footprint and iOS headroom before admission;
this cleanup never runs inside a forward or between a cold/warm pair. Reports
retain before/after samples, admitted components and failures for each phase.
The same boundary check covers initial dense-file preparation, so a first
installation cannot bypass this cleanup and phase reporting.

Prediction is optional I/O. When fewer slots fit, its depth falls before the
actual selected-expert allowance can fall below eight. If eight demand slots
do not fit, admission still fails. A zero-depth run is explicitly recorded and
displayed as having no active lookahead. The real header supplies a regression
for simulated 4.3 GB and 4.1 GB ceilings, with eight demand slots plus three or
zero prediction slots, respectively; a 4.0 GB example must still fail. These
are policy inputs, not measurements of the phone's actual available memory.
Physical confirmation of the change remains pending.

## Validated private build 113

[Run 39105887981](https://github.com/Dhafen/flash-local/actions/runs/39105887981)
delivered 0.1.0 (113), runtime source
`c3fe7be385271bcd33b61af51661bea880b4ea29`. Native CPU cache/lookahead/page-cache
controls remain bitwise equal to the resident CPU path for F32, mixed
Q4_K/Q6_K and 128 top-8 positions. Their native owners are released after
each context. The independent old CPU oracle differs by less than 5e-10.
Nine touched VM regions disappear after free; Swift closes and reopens three
native owners while retaining the old closed wrapper. All 18 selected Swift
functions and three-platform framework/header/import/isolation checks pass.
These controls do not execute trained Metal inference on iPhone.

The unchanged QPack controls, 580 app tests, 474 package invocations including
9 environment skips, ten synthetic ordinary Ornith logit comparisons and
all 36 actually reviewed Mac/iOS captures pass. The arm64 archive, signature,
memory entitlements, privacy manifest and no-bundled-model checks pass.
Private upload and a fresh Apple audit confirm the exact unchanged sole-owner
group is eligible. The iOS runtime is the source above; subsequent documentation
does not change it. Local report-analysis tests also pass (24 cases).

The ordinary chat still uses its existing runner. Native LES remains an
opt-in comparison, with no model redownload. The next phone report must
establish recovered phase-boundary memory, the trained routing/logit gate
and comparable 128-token performance. No LES throughput, full trained-model
quality or fastest-backend result has yet been verified on the phone.

## Validated private build 112

[Run 39073532467](https://github.com/Dhafen/flash-local/actions/runs/39073532467)
delivered version 0.1.0 (112) from
`207b5334df8661f62222af954717fb041a01311d`. The native controls, 18 selected
Swift tests (10 LES and 8 tokenizer/reasoning tests), QPack controls, 580 app
tests and 474 package tests passed; the package run had 9 environment skips.
All 36 exported iOS Simulator/macOS captures were reviewed in light and dark
appearances. The arm64 device archive, signing, memory entitlements, privacy
checks and internal-only delivery also passed. A fresh Apple audit confirmed
availability to the unchanged sole-owner private group.

The build includes streamed thinking and the adaptive memory admission above.
The ordinary chat still uses its existing runner; LES is an opt-in comparison.
The subsequent phone report stops at the second native admission as detailed
above. Trained Metal/reference agreement, comparable 128-token performance
and a fastest-backend result remain pending.
Hosted tests and reviewed simulator captures do not establish these results.

## Local analysis of phone exports

### Build 113 trained phone gate

The subsequent physical report admits all three numerical variants. After
the first native owner closes, the transition footprint is 238,160,024 bytes;
before batched prefill it is 244,664,496 bytes. Both samples report zero live
native contexts, buffers, owned bytes and mapped spans. Device allocation is
2,228,224 bytes. This is evidence of recovery in that run, not a larger cache
allowance or a general memory-pressure guarantee.

Prediction off/on have identical logit errors and route digest/count:
maximum absolute error 3.11909437, relative L2 0.06595874, 8,000 decisions.
Prefill-four has maximum error 3.30069351 and relative L2 0.08927649. All four
greedy choices match, but the unchanged 0.01 absolute / 0.001 relative limits
correctly prevent timing. The report contains no speed measurements. Its
legacy batch digest cannot establish batch routing because of the observer
stride/order limitation addressed above. Low Power Mode was enabled, so a
future timing comparison also needs that mode disabled.

The new opt-in diagnostic retains at most 64 MiB of fixed-test checkpoint
payload, with a 4 MiB/256-point per-forward cap. Both runtimes observe known
embedding, attention, FFN and layer outputs; detailed QKV/convolution points
are limited to layer zero. The raw layer-zero DeltaNet result (stage 10)
is distinct from the final attention projection (stage 5): the upstream
base class and model graph reuse `attn_output` for different operations.
No point is silently overwritten or accepted as a duplicate. The newer
graph's `h_pre_norm` alias maps to the actual last main-layer output, excluding
the stored MTP block. Early native controls require all 25 semantic points
in each generated four-layer forward and after reset, alongside unchanged
full-vocabulary logits. Only scalar errors, missing/shape counts and the
first divergent position/stage/layer are exported. Raw activation vectors
remain ephemeral. A first-layer, 16-row matched-input probe compares observed
and ordinary CPU results independently with a dequantized-weight double-dot
reference. It is a sampled operator diagnostic, not a replacement model
reference or permission to relax the full-logit gate. One hypothesis is CPU
operand rounding versus Metal's operand path; a cause is not yet established.
Timing runs do not enable either observer. Weights, top-8, ordinary chat,
prepared file format and acceptance thresholds remain unchanged. Build 119
passes Apple build/delivery validation; trained device evidence remains pending.

The original CPU/native/prefill-four checks run without extra layer or
projection observation. Only after their gate fails is a separate diagnostic
replay performed, using the original fixed prompt and the same forced tokens.
Both CPU and native logit changes caused by observation are measured. A good
instrumented result cannot admit timing or replace the failed original check;
an unavailable diagnostic still preserves the original failure report.
The original native admission checks still observe routing. A separate
comparison of callback-free timing execution is also required before
promoting a backend; the four-position gate is not full quality verification.

The routing observer reads each top-8 row with the tensor's actual row
stride. In this pinned ggml, top-k is a view of a full expert sort, so batched
rows are not necessarily packed consecutively. The expert loader already
honours that stride; this corrects diagnostics without changing model routing.
A new canonical checksum includes token position, layer, rank and selected
expert. It permits comparison across token-major decode and layer-major
prefill. Missing, partial or differing new checksums cannot admit a native
benchmark. Legacy reports retain their original short gate and explicitly
lack this additional batch-routing evidence.

Native host controls now replay uneven prompt prefixes with batch size four,
then hand off to single-token decode through 128 teacher-forced positions.
Resident CPU logits, greedy choices and canonical routes must agree with
cache, lookahead and page-cache modes. Batch size four was already used by
the shipped native benchmark. These are additional correctness controls;
they do not establish a new phone speed or a new TestFlight delivery.

[Native-only run 39111809758](https://github.com/Dhafen/flash-local/actions/runs/39111809758)
passed at source `e601d712c9110bbc54bcbeed79fb11877d56ba31`. All 63
prefix/mode variants have zero maximum logit difference from resident CPU,
matching greedy choices and canonical routes. F32 and mixed Q4_K/Q6_K use
five positions; the 256-expert/top-8 graph uses 128 teacher-forced positions.
The independent old CPU oracle remains within 5e-10. All 18 selected Swift
functions (10 LES and 8 tokenizer/thinking), the ownership/VM tests and
three Apple framework/header/import/export slices pass. Local report analyzers
pass 26 tests. This validates the isolated native changes, not a new full-app
archive or TestFlight upload. Build 113 remained the shipped runtime at that
point.

[Full delivery run 39130511069](https://github.com/Dhafen/flash-local/actions/runs/39130511069)
passes at runtime source `8792fd52ea525fafec8145dbc3aa4b0195cf28d6`, delivered
as private TestFlight 119. All 63 batch variants still have zero logit error.
Additional native F32/mixed/top-8 contexts require all 25 semantic points on
every four-layer forward and after reset, with zero logit difference from
resident CPU; the top-8 case uses 128 teacher-forced positions. Both old CPU
wrapper controls retain exact logits over five forwards with 125 points.
The matched-input projection control, all 19 selected Swift functions,
three Apple slices, 186 Swiftlet/24 QPack tests, app/package gates, 36 reviewed
captures, device archive, signing and private availability pass. Local LES
report analysis passes 19 tests. These are synthetic/metadata/build results;
the fastest backend remains unverified.

The subsequent physical build-119 export contains 5,125 compared semantic
points (205 per forward across 25 forwards), with zero missing points or shape
mismatches. Embeddings match exactly. Layer-zero attention normalization has
maximum absolute error 5.72e-6; the first threshold violation is QKV at position
zero, with 0.05444 maximum error and 0.004192 relative L2. Added layer observers
change neither runtime's four output logit vectors in this replay. The 16-row
matched-input Q4_K projection probe has maximum Metal error 1.57e-7 versus
dequantized-weight/F32 dots, compared with 0.009567 for the ordinary CPU's
Q8_K operand path. This establishes the arithmetic difference for that sample,
not full-model correctness. Single-token off/on logits still have 0.06596
relative L2 error against ordinary CPU; batch-four has 0.08928, and its canonical
route checksum differs. Both supplied files contain the same report. No speed
measurements completed. Sampled peak was 2.288 GB; all four phase boundaries
have zero owned native contexts/buffers. Airplane mode was user-declared.

An opt-in third reference is validated separately. It keeps the old
source pin, model graph and original GGUF bytes, and replaces every model-weight
linear product at a synchronized scheduler boundary with row dequantization
and a double-accumulated dot using the original F32 activations. Dense and
selected-expert paths are covered; real expert IDs and padded row strides are
preserved. Linear scratch is capped at one MiB and governor checks occur at
least every 64 output rows. The default CPU reference, numerical thresholds,
production GGUF/REAM runner are unchanged. This is
not a performance backend and does not prove that native batch-four is correct.
New host controls compare it with an independent all-F32 graph made only from
the same bounded synthetic fixture's dequantized weights. No trained model is
downloaded or converted on the build host, and the phone reference never writes
an expanded model. [Native-only run 39136311025](https://github.com/Dhafen/flash-local/actions/runs/39136311025)
passes at `2a0e2c497b63e23344a23e5b12145d5bca936ede`: both five-forward
full-vocabulary graph controls are within 9.32e-8 maximum absolute and 1.64e-7
relative L2 error of the independently executed F32 graphs. The eight-format
operator controls, unchanged original CPU replay, all 19 selected Swift
functions and three Apple slices pass. These synthetic results validate that
initial oracle implementation, not trained phone inference or the subsequent
integration changes.

The build-120 integration preserves the original `.passed` CPU comparison
and reports an explicitly named `arithmeticCheck`. Native timing requires
this separate, complete higher-precision reference, identical canonical
position/layer/rank/expert checksums against the CPU oracle across every replay
token, the unchanged 0.01 absolute / 0.001 relative logit limits, and successful
callback-free checks of cache, lookahead and page-cache modes. A projection
sample or an observed-only diagnostic cannot satisfy this admission. Serial
prefill remains available when batch-four fails its independent logit/routing
or callback-free checks; actual batch size is recorded in each measurement.
This retains all model weights and top-8 choices. Only prompt scheduling varies.
The new per-context reference routing observer shares the unchanged canonical
hash implementation with LES. Its linear scratch is a single lazy 786,432-byte
allocation, independent of column-growth history. Failed arithmetic checks
retain bounded layer diagnostics against that same higher-precision oracle.
The original chat is never promoted or changed by any benchmark result.

## Build 120 phone evidence

The four supplied files identify runtime source `1f3fbebd77bc-dirty` and build
120. Their entire numerical checks are identical; two files are byte-identical.
Only the first contains fresh admission samples and measured peak footprint
(2,302,268,280 bytes). The later exports reused a cached failed check and contain
no new admissions or observed footprint. They are not four independent runs.
All reports contain zero timing measurements. The test worker now performs a
fresh reference check and memory sampling for each explicit run rather than
re-exporting a cached failed result. Existing verified weights/preparation stay
reusable.

Against the new direct dequantized-weight/F32 reference, the actual serial
native paths have maximum absolute logit error 0.00377691, relative L2
0.0000444115 and matching four greedy choices. Cache, lookahead and page-cache
callback-free paths report these same errors. This satisfies the unchanged
0.01 / 0.001 logit bounds, but does not establish full model quality. Across
25 forwards and 8,000 expert decisions, the reference ranked checksum is
4421523720827196693 and serial native is 2129472169425831000. Native lookahead
on/off ranked checksums match each other exactly. Build 120 cannot distinguish
expert rank permutations within the same eight from actual selection changes;
the check therefore correctly stops timing.

Batch-four has 2.28847 maximum logit error and 0.0592567 relative L2 against
the same independent reference. It remains unadmitted. Bounded layer tracing
compares all 5,125 semantic points with no missing points or shape mismatch;
the first layer threshold crossing is post-attention normalization in layer
31 at position one (0.0100813 absolute). Both reference/native observation
effects are exactly zero on the four output vectors. The original ordinary
CPU comparison is still reported, including its roughly 0.06596 relative L2
difference; changing the reference is not an engine speed or quality gain.

The next isolated revision adds a second checksum over each absolute
position/layer/expert tuple, omitting only rank. Duplicate expert IDs are
rejected before committing either checksum. It never sorts or modifies model
data. The explicitly named `exact-expert-set-v1` admission requires complete
identical expert sets against the independent reference, the unchanged strict
full-vocabulary and callback-free limits, and still requires ranked native
lookahead on/off traces to be identical. An altered expert, incomplete trace,
unknown policy or missing set proof cannot pass. The original ranked checksums
remain exported. Rank permutations can change floating-point reduction order
while preserving the mathematical expert-weight pairs; only that separately
verified case is accepted. Failed batch checks select independently checked
serial prefill, without changing top-k, quantization or weights. Host tests
are being run; trained expert-set agreement and phone throughput remain open.

Run `python3 scripts/analyze_ornith_les.py REPORT.json` on an exported LES
report. It only reads the local JSON and prints whitelisted aggregates.
Failed admissions retain before/after footprint and recover published numeric
reservations without echoing raw errors, paths or chat content. A sampled peak
alone is never substituted for the live baseline at rejection.

Each cache state receives a provisional, single-prompt ranking only when all
four modes complete at least 128 tokens, the short numerical/routing gate
passes, token-output hashes and prompt lengths agree, memory stays in budget
and thermal conditions are comparable. Zero-depth lookahead, inconsistent
mode geometry, partial GPU timestamps and missing data are reported explicitly.
The analysis does not select the app's backend or establish full model quality,
physical flash traffic, authenticated offline execution or sustained performance.
