#ifndef FLASH_LLAMA_REFERENCE_H
#define FLASH_LLAMA_REFERENCE_H
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Diagnostic CPU reference only. No GPU backend, mlock, weight repacking or
// production streaming guarantee. The caller supplies a live memory/cancel
// governor which remains alive until flr_free returns. A true callback aborts.
typedef bool (*flr_abort_callback)(void * user);
typedef struct flr_context flr_context;
flr_context * flr_create(int verified_descriptor, int32_t context_tokens,
                         flr_abort_callback should_abort, void * user);
// Opt-in bounded checkpoint observer for the fixed local numerical test.
// stage IDs: 1 embedding, 2 attention norm, 3 first-layer QKV, 4 first-layer
// convolution, 5 attention output, 6 post-attention norm, 7 FFN output,
// 8 layer output, 9 final norm. Values are transient and must not escape the
// callback without a caller-owned bounded copy. False aborts this diagnostic.
typedef bool (*flr_checkpoint_callback)(uint32_t stage, int32_t layer,
    const float * values, int32_t count, void * user);
flr_context * flr_create_traced(int verified_descriptor, int32_t context_tokens,
    flr_abort_callback should_abort, void * user, flr_checkpoint_callback, void * checkpoint_user);
// Independent diagnostic third oracle. Keeps the pinned graph and original
// packed weights; every model-weight MUL_MAT/MUL_MAT_ID uses row dequantization
// and a double-accumulated dot with the original F32 activation. The ordinary
// CPU reference above is retained. Optional checkpoints; <=1 MiB linear scratch.
// Ornith only, serial short replay, never a production/performance backend.
flr_context * flr_create_dequantized(int verified_descriptor, int32_t context_tokens,
    flr_abort_callback should_abort, void * user, flr_checkpoint_callback, void * checkpoint_user);
typedef struct {
    uint64_t dense_calls, expert_calls, rows, maximum_scratch_bytes;
} flr_dequantized_metrics;
flr_dequantized_metrics flr_dequantized_snapshot(const flr_context *);
void flr_free(flr_context *);
const char * flr_last_error(void);
const char * flr_source_revision(void);
int32_t flr_vocabulary_size(const flr_context *);
// One explicit token, one full vocabulary of logits. A failed decode poisons the
// context: release it rather than resuming from partially advanced state.
int32_t flr_step(flr_context *, int32_t token, float * output, int32_t capacity);
// Production projection helper: canonical CPU activation rounding only. No
// model/context/file is opened. One bounded row; caller owns reusable buffers.
int32_t flr_round_linear_activations(uint32_t weight_type, const float * input,
    int32_t elements, float * output, int32_t output_capacity,
    void * packed_scratch, int32_t packed_capacity_bytes);
// Production small K/V projection only: <=1024 output rows, <=65536 columns,
// <=4 MiB packed weights. Uses pinned CPU operand rounding and vec_dot, without
// a model/context/file or heap weight copy. All scratch belongs to the caller.
int32_t flr_project_kv_rows(uint32_t weight_type, const void * weights, int32_t weight_bytes,
    const float * input, int32_t columns, int32_t rows, float * output, int32_t output_capacity,
    float * rounding_scratch, int32_t rounding_capacity, void * packed_scratch, int32_t packed_capacity);
#ifdef __cplusplus
}
#endif
#endif
