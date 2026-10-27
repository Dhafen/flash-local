#pragma once
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct flf_context flf_context;
typedef bool (*flf_abort)(void *);
// Fraction is -1 for an unmeasurable phase; loader fractions are real events.
// Phases: 0 metadata/admission, 1 weights, 2 KV/graph, 3 ready.
typedef void (*flf_progress)(int32_t phase, float fraction, void *);
typedef struct {
    int32_t context_tokens, evaluation_batch, physical_batch, cpu_threads;
    uint64_t process_ceiling_bytes, app_reserve_bytes;
    bool metal, f32_kv_control;
    bool diagnostic_routing; // installs a graph callback; false for serving
} flf_options;
typedef struct {
    // Explicit expert pread totals only; not mmap faults/physical NAND traffic.
    uint64_t io_bytes, io_calls, cached_tokens, evaluated_tokens, state_only_tokens;
    uint64_t estimated_process_bytes, expert_bank_bytes, estimated_kv_bytes;
    double forward_seconds;
    bool flash_attention_active;
} flf_metrics;
// Caller verifies the complete file SHA and holds an exclusive APP engine
// lease (also excluding Ornith). The file descriptor is retained by duplication.
// Directory is an empty private session directory. No weights are copied;
// only a disposable expert-offset manifest is written there. Calls except
// cancel require serial ownership. Close only after in-flight calls return.
flf_context * flf_open(int verified_fd, const char * private_session_directory,
                       flf_options, flf_abort, flf_progress, void *);
void flf_close(flf_context *);
void flf_cancel(flf_context *); // may be called from another thread
int flf_reset(flf_context *);  // clears KV/position and cancellation, retains weights
int32_t flf_vocab_size(const flf_context *);
int flf_tokenize(flf_context *, const char * utf8, int32_t bytes, int32_t * tokens, int32_t capacity);
int flf_token_piece(flf_context *, int32_t token, char * utf8, int32_t capacity);
// Publisher's Rec template, including reasoning replay and named XML tool args.
// Input contains messages/tools/add_generation_prompt. Negative = bytes needed.
int flf_render_prompt(const char * json, int32_t bytes, char * utf8, int32_t capacity);
bool flf_is_eog(flf_context *, int32_t token);
// No logits during intermediate prefill. Final prompt/decode needs full logits.
int flf_evaluate(flf_context *, const int32_t * tokens, int32_t count,
                  float * final_logits, int32_t capacity);
// Production decode keeps the vocabulary row in native memory. No 150k-float
// copy or Swift vocabulary sort per token. copy_logits is diagnostic only.
int flf_evaluate_for_sampling(flf_context *, const int32_t * tokens, int32_t count);
int flf_copy_logits(flf_context *, float * logits, int32_t capacity);
// Seeded GGML sampling; temperature 0 is greedy. top_k=0 retains all tokens.
// This vocabulary top-k is unrelated to Flint's immutable top-6 MoE routing.
int flf_configure_sampling(flf_context *, float temperature, float top_p, int32_t top_k, uint32_t seed);
int32_t flf_sample(flf_context *); // -1 on error; requires a final logit row
// Optional bounded routing diagnostics; off by default. Never used for expert
// prediction. Toggle clears the trace. Negative = JSON buffer bytes needed.
int flf_trace_routes(flf_context *, bool enabled);
int flf_routing_json(flf_context *, char * utf8, int32_t capacity);
flf_metrics flf_snapshot(const flf_context *);
const char * flf_last_error(void);
const char * flf_revision(void);
const char * flf_library_path(void); // local integrity identity, not diagnostics
#ifdef __cplusplus
}
#endif
