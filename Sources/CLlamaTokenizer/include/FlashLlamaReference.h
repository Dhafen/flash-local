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
#ifdef __cplusplus
}
#endif
#endif
