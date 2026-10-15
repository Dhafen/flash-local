#ifndef FLASH_LLAMA_TOKENIZER_H
#define FLASH_LLAMA_TOKENIZER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct flt_vocabulary flt_vocabulary;
// Caller keeps its descriptor. The bridge duplicates it, loads vocab_only,
// closes its FILE, and never creates an inference context or reads weights.
flt_vocabulary * flt_load_vocabulary(int descriptor);
void flt_free_vocabulary(flt_vocabulary * vocabulary);
const char * flt_last_error(void);
int32_t flt_vocabulary_size(const flt_vocabulary * vocabulary);
int32_t flt_eos(const flt_vocabulary * vocabulary);
int32_t flt_eot(const flt_vocabulary * vocabulary);
bool flt_is_eog(const flt_vocabulary * vocabulary, int32_t token);
const char * flt_chat_template(const flt_vocabulary * vocabulary);
// Nonnegative = written length; negative = required capacity. INT32_MIN is
// an invalid request/failure. Null output with capacity zero is a size query.
int32_t flt_encode(const flt_vocabulary *, const char * text, int32_t bytes,
                   bool parse_special, int32_t * output, int32_t capacity);
int32_t flt_token_piece(const flt_vocabulary *, int32_t token, bool special,
                        char * output, int32_t capacity);
// JSON array of typed text messages. Fixed no-thinking/no-tools policy;
// validates roles and rejects structural-token injection in message text.
int32_t flt_render_chat(const flt_vocabulary *, const char * messages_json,
                        int32_t bytes, char * output, int32_t capacity);
#ifdef __cplusplus
}
#endif
#endif
