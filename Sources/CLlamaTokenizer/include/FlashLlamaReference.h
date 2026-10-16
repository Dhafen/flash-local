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
