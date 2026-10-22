#pragma once
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct flo_context flo_context;
typedef bool (*flo_abort)(void *);
// Optional phase-only diagnostics. No tensors, token IDs or paths cross this
// callback. False stops before the next native operation. Disabled for timing.
typedef bool (*flo_diagnostic)(int32_t phase, void *);
typedef bool (*flo_checkpoint)(uint32_t stage, int32_t layer, const float * values, int32_t count, void *);
typedef struct {
    uint64_t io_bytes, io_calls, misses, selected_experts;
    uint64_t predicted, prediction_hits, prefetch_bytes;
    uint64_t process_ns, install_ns, dispatches, command_buffers, gpu_ns;
    uint64_t routing_digest, routing_decisions;
    uint64_t demand_hits, demand_misses;
    uint64_t timestamped_command_buffers;
    // Process-global LES ownership. Mapped spans/device allocations are not
    // physical footprint. Device bytes may include other app Metal users.
    uint64_t live_contexts, live_metal_buffers, owned_metal_bytes, mapped_metal_bytes, device_allocated_bytes;
    bool metal_device_observed;
    uint64_t canonical_routing_digest;
    uint64_t probe_samples, probe_columns, probe_weight_type, probe_cpu_input_type;
    double probe_backend_max, probe_cpu_max, probe_backend_squared, probe_cpu_squared, probe_reference_squared;
    uint64_t canonical_expert_set_digest;
    uint64_t configured_physical_batch_tokens, maximum_physical_batch_tokens;
    uint64_t skipped_intermediate_output_batches;
    uint64_t target_tokens_evaluated, dflash_blocks, dflash_proposals, dflash_accepted, dflash_replays;
    uint64_t dflash_feature_peak_bytes;
    bool flash_attention_active;
    uint64_t state_only_prefill_tokens, state_only_prefill_calls;
} flo_metrics;
typedef struct {
    int32_t context_tokens, prefill_slots, cache_slots_per_layer;
    int32_t lookahead_depth, io_threads, cpu_threads;
    bool metal, bypass_page_cache, trace_routing;
    uint64_t memory_ceiling_bytes;
    // Experimental benchmark contexts only. Ordinary chat defaults stay zero.
    uint64_t additional_reserve_bytes;
    bool flash_attention, experimental_dflash;
    // Opt-in fixed-test checkpoints; stage IDs match FlashLlamaReference.h.
    // Borrowed values live only during the callback. Never enabled for timing.
    flo_checkpoint checkpoint;
    void * checkpoint_user;
    flo_diagnostic diagnostic;
    void * diagnostic_user;
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
// Evaluate every token and retain KV/SSM, without materializing vocabulary
// logits. Synchronizes before returning; the final prompt call uses flo_decode.
int flo_ingest(flo_context *, const int32_t * tokens, int32_t count);
void flo_reset(flo_context *);
// One bounded recurrent-only checkpoint. Attention KV remains in place and is
// trimmed on restore. No model weights or full-context KV copy is made.
uint64_t flo_capture_prefix(flo_context *);
int flo_restore_prefix(flo_context *);
// Benchmark-only greedy block verification. Reuses the sole recurrent snapshot;
// must not be combined with a live conversation checkpoint. Returns committed
// anchor + matching proposals; rejected suffix never escapes. Restores GDN/conv
// and attention state before returning on rejection.
int flo_verify_greedy(flo_context *, const int32_t * block, int32_t count,
                      float * next_logits, int32_t vocabulary);
int flo_attach_dflash(flo_context *, int verified_draft_fd);
int flo_dflash_step(flo_context *, float * current_logits, int32_t vocabulary,
                    int32_t * committed_tokens, int32_t maximum_tokens);
flo_metrics flo_snapshot(const flo_context *);
const char * flo_last_error(void);
const char * flo_revision(void);
#ifdef __cplusplus
}
#endif
