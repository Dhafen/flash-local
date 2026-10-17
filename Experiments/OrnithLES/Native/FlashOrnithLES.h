#pragma once
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct flo_context flo_context;
typedef bool (*flo_abort)(void *);
typedef struct {
    uint64_t io_bytes, io_calls, misses, selected_experts;
    uint64_t predicted, prediction_hits, prefetch_bytes;
    uint64_t process_ns, install_ns, dispatches, command_buffers, gpu_ns;
    uint64_t routing_digest, routing_decisions;
    uint64_t demand_hits, demand_misses;
    uint64_t timestamped_command_buffers;
} flo_metrics;
typedef struct {
    int32_t context_tokens, prefill_slots, cache_slots_per_layer;
    int32_t lookahead_depth, io_threads, cpu_threads;
    bool metal, bypass_page_cache, trace_routing;
    uint64_t memory_ceiling_bytes;
} flo_options;
// Input is a caller-verified open regular file, not a path that can be replaced.
// Writes only a small dense GGUF and a manifest. Routed weights stay in input.
// The output directory must be empty and owned by the caller. No quantization.
int flo_prepare(int verified_fd, const char * output_directory, flo_abort, void *);
flo_context * flo_open(int verified_fd, const char * prepared_directory,
                      flo_options, flo_abort, void *);
void flo_close(flo_context *);
int32_t flo_vocabulary_size(const flo_context *);
// All tokens are evaluated; only the final position's full logits are returned.
int flo_decode(flo_context *, const int32_t * tokens, int32_t count,
               float * logits, int32_t capacity);
void flo_reset(flo_context *);
flo_metrics flo_snapshot(const flo_context *);
const char * flo_last_error(void);
const char * flo_revision(void);
#ifdef __cplusplus
}
#endif
