#include "FlashLlamaReference.h"
#include "llama.h"
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
