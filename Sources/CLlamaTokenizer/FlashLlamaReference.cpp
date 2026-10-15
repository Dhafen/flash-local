#include "FlashLlamaReference.h"
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>

struct flr_context {
    FILE * file = nullptr;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    flr_abort_callback should_abort = nullptr;
    void * user = nullptr;
    int32_t vocabulary = 0, position = 0, limit = 0;
    bool poisoned = false;
    ~flr_context() {
        if (context) llama_free(context);
        if (model) llama_model_free(model);
        if (file) fclose(file);
    }
};
static thread_local std::string reference_error;
static std::once_flag backend_once;
static bool aborted(flr_context * value) { return value->should_abort(value->user); }
static bool loading(float, void * user) { return !aborted(static_cast<flr_context *>(user)); }
static bool decoding(void * user) { return aborted(static_cast<flr_context *>(user)); }
const char * flr_last_error() { return reference_error.c_str(); }
const char * flr_source_revision() { return FLASH_LLAMA_REVISION; }

int32_t flr_round_linear_activations(uint32_t weight_type, const float * input,
    int32_t elements, float * output, int32_t capacity, void * packed, int32_t packed_bytes) {
    try {
        switch (weight_type) {
            case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
            case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q4_K: case GGML_TYPE_Q6_K: break;
            default: reference_error = "Unsupported projection weight type"; return -1;
        }
        if (!input || !output || !packed || elements < 1 || elements > 65536 ||
            capacity < elements || packed_bytes < 1 || packed_bytes > 131072 ||
            uintptr_t(input) % alignof(float) || uintptr_t(output) % alignof(float) || uintptr_t(packed) % 16) {
            reference_error = "Activation rounding buffer bound/alignment"; return -2;
        }
        const auto overlaps = [](const void * a, size_t an, const void * b, size_t bn) {
            const uintptr_t x = uintptr_t(a), y = uintptr_t(b);
            return x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn || (x < y + bn && y < x + an);
        };
        const size_t float_bytes = size_t(elements) * sizeof(float);
        if (overlaps(input, float_bytes, output, float_bytes) ||
            overlaps(input, float_bytes, packed, size_t(packed_bytes)) ||
            overlaps(output, float_bytes, packed, size_t(packed_bytes))) {
            reference_error = "Activation rounding buffers overlap"; return -2;
        }
        for (int32_t i = 0; i < elements; ++i) if (!std::isfinite(input[i])) {
            reference_error = "Nonfinite projection activation"; return -2;
        }
        if (weight_type == GGML_TYPE_F32) { std::copy_n(input, elements, output); return 0; }
        std::call_once(backend_once, [] { llama_backend_init(); });
        const auto activation = ggml_get_type_traits_cpu(ggml_type(weight_type))->vec_dot_type;
        if (elements % ggml_blck_size(activation) ||
            ggml_row_size(activation, elements) > size_t(packed_bytes)) {
            reference_error = "Activation rounding block/scratch bound"; return -2;
        }
        const auto from = ggml_get_type_traits_cpu(activation)->from_float;
        const auto to = ggml_get_type_traits(activation)->to_float;
        // Q8_K is an intermediate CPU operand and has no public to_float trait;
        // call its unchanged canonical dequantizer directly.
        if (!from || (!to && activation != GGML_TYPE_Q8_K)) {
            reference_error = "Canonical activation conversion unavailable"; return -1;
        }
        if (activation == GGML_TYPE_Q8_0 || activation == GGML_TYPE_Q8_K) {
            const int32_t block = int32_t(ggml_blck_size(activation));
            for (int32_t start = 0; start < elements; start += block) {
                float maximum = 0;
                for (int32_t i = 0; i < block; ++i) maximum = std::max(maximum, std::fabs(input[start + i]));
                if (maximum > 0 && !std::isfinite(127.f / maximum)) {
                    reference_error = "Activation quantization scale is not representable"; return -2;
                }
            }
        }
        from(input, packed, elements);
        if (activation == GGML_TYPE_Q8_K) {
            dequantize_row_q8_K(static_cast<const block_q8_K *>(packed), output, elements);
        } else { to(packed, output, elements); }
        for (int32_t i = 0; i < elements; ++i) if (!std::isfinite(output[i])) {
            reference_error = "Nonfinite rounded projection activation"; return -2;
        }
        return 0;
    } catch (const std::exception & error) { reference_error = error.what(); return -3; }
}

flr_context * flr_create(int descriptor, int32_t context_tokens, flr_abort_callback callback, void * user) {
    try {
        if (!callback || context_tokens < 16 || context_tokens > 128) {
            reference_error = "CPU diagnostic requires a governor and 16...128 context tokens"; return nullptr;
        }
        auto value = std::make_unique<flr_context>();
        value->should_abort = callback; value->user = user; value->limit = context_tokens;
        if (aborted(value.get())) { reference_error = "CPU reference cancelled before loading"; return nullptr; }
        const int duplicate = fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
        if (duplicate < 0) { reference_error = "Cannot duplicate verified CPU reference descriptor"; return nullptr; }
        value->file = fdopen(duplicate, "rb");
        if (!value->file) { close(duplicate); reference_error = "Cannot open CPU reference FILE"; return nullptr; }
        if (fseek(value->file, 0, SEEK_SET)) { reference_error = "Cannot seek CPU reference file"; return nullptr; }
        std::call_once(backend_once, [] { llama_backend_init(); });
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode = LLAMA_LAZY_MODE_ON; mp.load_mtp = false;
        mp.use_extra_bufts = false; mp.check_tensors = false;
        mp.progress_callback = loading; mp.progress_callback_user_data = value.get();
        value->model = llama_model_load_from_file_ptr(value->file, mp);
        if (!value->model || aborted(value.get())) { reference_error = "CPU model load failed or governor stopped it"; return nullptr; }
        char architecture[32] = {};
        llama_model_meta_val_str(value->model, "general.architecture", architecture, sizeof(architecture));
        if (std::string(architecture) != "qwen4exp") { reference_error = "CPU diagnostic requires qwen4exp"; return nullptr; }
        value->vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(value->model));
        if (value->vocabulary < 1 || value->vocabulary > 300000) { reference_error = "CPU vocabulary exceeds output bound"; return nullptr; }
        auto cp = llama_context_default_params();
        cp.n_ctx = uint32_t(context_tokens); cp.n_batch = 1; cp.n_ubatch = 1; cp.n_seq_max = 1;
        cp.n_threads = 2; cp.n_threads_batch = 2;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = false; cp.op_offload = false; cp.moe_cache_size = 0;
        cp.abort_callback = decoding; cp.abort_callback_data = value.get();
        value->context = llama_init_from_model(value->model, cp);
        if (!value->context || aborted(value.get())) { reference_error = "CPU context allocation failed or governor stopped it"; return nullptr; }
        reference_error.clear(); return value.release();
    } catch (const std::exception & e) { reference_error = e.what(); return nullptr; }
    catch (...) { reference_error = "Unknown CPU reference load failure"; return nullptr; }
}

void flr_free(flr_context * value) { delete value; }
int32_t flr_vocabulary_size(const flr_context * value) { return value ? value->vocabulary : -1; }
int32_t flr_step(flr_context * value, int32_t token, float * output, int32_t capacity) {
    try {
        if (!value || value->poisoned || token < 0 || token >= value->vocabulary || !output
            || capacity != value->vocabulary || value->position >= value->limit) {
            reference_error = "CPU reference token, output capacity or state invalid"; return -1;
        }
        value->poisoned = true;
        if (aborted(value) || llama_decode(value->context, llama_batch_get_one(&token, 1)) || aborted(value)) {
            reference_error = "CPU decode failed or governor stopped it"; return -1;
        }
        const float * logits = llama_get_logits_ith(value->context, -1);
        if (!logits || !std::all_of(logits, logits + value->vocabulary, [](float x) { return std::isfinite(x); })) {
            reference_error = "CPU reference produced missing or nonfinite logits"; return -1;
        }
        std::copy_n(logits, value->vocabulary, output);
        value->position++; value->poisoned = false; reference_error.clear(); return value->vocabulary;
    } catch (const std::exception & e) { reference_error = e.what(); return -1; }
    catch (...) { reference_error = "Unknown CPU reference decode failure"; return -1; }
}
