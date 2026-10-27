#include "FlashFlint.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-moe-slot.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;
static thread_local std::string error_text;
void flint_set_error(const std::string & value) { error_text = value; }
static std::atomic<bool> leased{false};
static std::once_flag initialized;
static void require(bool ok, const char * text) { if (!ok) throw std::runtime_error(text); }
struct flf_context {
    int source = -1;
    FILE * stream = nullptr;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;
    std::vector<llama_token_data> candidates;
    std::string manifest;
    struct stat identity{};
    flf_options options{};
    flf_metrics metrics{};
    std::atomic<bool> cancelled{false};
    flf_abort abort = nullptr;
    flf_progress progress = nullptr;
    void * user = nullptr;
    bool failed = false;
    bool logits_valid = false;
    bool trace_routes = false;
    struct route { int32_t position, layer; std::array<int32_t, 6> experts; };
    std::vector<route> routes;
    std::array<uint32_t, 49> trace_layer_rows{};
    int32_t layers = 0, trace_final_position = -1;
    bool trace_output_requested = false;
    int32_t slots = 0;
    ~flf_context() {
        if (sampler) llama_sampler_free(sampler);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
        if (stream) fclose(stream);
        if (source >= 0) close(source);
        if (!manifest.empty()) unlink(manifest.c_str());
        leased.store(false);
    }
    bool should_abort() const { return cancelled.load() || (abort && abort(user)); }
    void check() const {
        require(!should_abort(), "Flint operation cancelled");
        struct stat now{};
        bool same = fstat(source, &now) == 0 && now.st_dev == identity.st_dev && now.st_ino == identity.st_ino &&
            now.st_size == identity.st_size && now.st_mtime == identity.st_mtime && now.st_ctime == identity.st_ctime;
#if defined(__APPLE__)
        same = same && now.st_mtimespec.tv_nsec == identity.st_mtimespec.tv_nsec && now.st_ctimespec.tv_nsec == identity.st_ctimespec.tv_nsec;
#else
        same = same && now.st_mtim.tv_nsec == identity.st_mtim.tv_nsec && now.st_ctim.tv_nsec == identity.st_ctim.tv_nsec;
#endif
        require(same,
            "Verified Flint file changed; re-verification required");
    }
};
static uint64_t rounded(uint64_t n) { return (n + 255) / 256 * 256; }
static uint32_t u32(const gguf_context * file, const char * name) {
    const auto key = gguf_find_key(file, name);
    require(key >= 0 && gguf_get_kv_type(file, key) == GGUF_TYPE_UINT32, "Missing Flint integer metadata");
    return gguf_get_val_u32(file, key);
}
static uint64_t add(uint64_t a, uint64_t b) {
    require(a <= std::numeric_limits<uint64_t>::max() - b, "Flint admission integer overflow");
    return a + b;
}
static void prepare_manifest(flf_context & c, const char * directory) {
    require(directory && fs::is_directory(directory) && fs::is_empty(directory), "Flint needs an empty private session directory");
    c.check();
    require(fseeko(c.stream, 0, SEEK_SET) == 0, "Cannot rewind verified Flint stream");
    ggml_context * tensors = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file_ptr(c.stream, {true, &tensors}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensor_owner(tensors, ggml_free);
    require(file && tensors && gguf_get_data_offset(file.get()) <= (16u << 20), "Invalid or excessive Flint metadata");
    const auto arch = gguf_find_key(file.get(), "general.architecture");
    require(arch >= 0 && gguf_get_kv_type(file.get(), arch) == GGUF_TYPE_STRING &&
        std::string(gguf_get_val_str(file.get(), arch)) == "flint", "Expected Flint GGUF architecture");
    const uint32_t layers = u32(file.get(), "flint.block_count"), dense_layers = u32(file.get(), "flint.leading_dense_block_count");
    c.layers = int32_t(layers);
    const uint32_t experts = u32(file.get(), "flint.expert_count"), topk = u32(file.get(), "flint.expert_used_count");
    const uint32_t nk = u32(file.get(), "flint.attention.head_count_kv"), head = u32(file.get(), "flint.attention.key_length");
    const uint32_t window = u32(file.get(), "flint.attention.sliding_window");
    require(layers > dense_layers && layers <= 49 && experts > 0 && experts <= 256 && topk == 6 && topk <= experts &&
        nk > 0 && nk <= 32 && head > 0 && head <= 256 && window > 1 && window <= 8192, "Unsupported Flint geometry");
    require(c.options.context_tokens <= int32_t(u32(file.get(), "flint.context_length")), "Requested context exceeds Flint metadata");
    const auto pattern = gguf_find_key(file.get(), "flint.attention.sliding_window_pattern");
    require(pattern >= 0 && gguf_get_kv_type(file.get(), pattern) == GGUF_TYPE_ARRAY && gguf_get_arr_n(file.get(), pattern) == layers,
        "Invalid Flint sliding-window pattern");
    const auto type = gguf_get_arr_type(file.get(), pattern);
    require(type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_BOOL, "Unsupported Flint pattern type");
    uint32_t full = 0;
    for (uint32_t il = 0; il < layers; ++il) {
        const uint32_t v = type == GGUF_TYPE_UINT32 ? static_cast<const uint32_t *>(gguf_get_arr_data(file.get(), pattern))[il] :
            static_cast<const bool *>(gguf_get_arr_data(file.get(), pattern))[il];
        require(v <= 1, "Invalid sliding-window layer flag"); full += v == 0;
    }
    const uint32_t slots = std::min(experts, topk * uint32_t(c.options.physical_batch));
    c.slots = slots;
    uint64_t dense_bytes = gguf_get_data_offset(file.get());
    json entries = json::array();
    std::map<std::pair<std::string, int>, uint64_t> bank_strides;
    std::regex routed("^blk\\.([0-9]+)\\.(ffn_(gate|up|down)_exps)\\.weight$");
    require(gguf_get_n_tensors(file.get()) <= 2000, "Excessive Flint tensor count");
    for (int64_t i = 0; i < gguf_get_n_tensors(file.get()); ++i) {
        c.check();
        const std::string name = gguf_get_tensor_name(file.get(), i);
        const auto * tensor = ggml_get_tensor(tensors, name.c_str());
        require(tensor && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_Q4_1), "Flint port admits original F32/Q4_1 only");
        const uint64_t size = ggml_nbytes(tensor);
        const uint64_t offset = add(gguf_get_data_offset(file.get()), gguf_get_tensor_offset(file.get(), i));
        require(offset <= uint64_t(c.identity.st_size) && size <= uint64_t(c.identity.st_size) - offset, "Flint tensor exceeds verified file");
        std::smatch match;
        if (!std::regex_match(name, match, routed)) { dense_bytes = add(dense_bytes, size); continue; }
        const uint32_t layer = std::stoul(match[1]);
        require(layer >= dense_layers && layer < layers && tensor->ne[2] == experts && tensor->ne[3] == 1 && size % experts == 0,
            "Flint expert geometry mismatch");
        auto key = std::make_pair(match[2].str(), int(tensor->type));
        const uint64_t stride = size / experts;
        require(!bank_strides.count(key) || bank_strides[key] == stride, "Unexpected mixed expert strides");
        bank_strides[key] = stride;
        entries.push_back({{"layer", layer}, {"tensor_family", match[2].str()}, {"tensor_name", name},
            {"quant_type", ggml_type_name(tensor->type)}, {"exact_byte_length", size}, {"bytes_per_expert", stride},
            {"expert_stride", stride}, {"expert_count", experts}, {"repacked_offset", offset}, {"repacked_file", "/dev/fd/" + std::to_string(c.source)}});
    }
    require(entries.size() == size_t(layers - dense_layers) * 3, "Incomplete Flint gate/up/down expert families");
    uint64_t bank = 0;
    for (const auto & group : bank_strides) bank = add(bank, group.second * slots);
    const uint64_t kv_cells = full * rounded(c.options.context_tokens) + (layers - full) * rounded(std::min<uint32_t>(
        c.options.context_tokens, window + c.options.physical_batch));
    c.metrics.estimated_kv_bytes = kv_cells * nk * head * 2 * (c.options.f32_kv_control ? 4 : 2);
    c.metrics.expert_bank_bytes = bank;
    c.metrics.estimated_process_bytes = add(add(add(dense_bytes, bank), c.metrics.estimated_kv_bytes),
        add(c.options.app_reserve_bytes, 1000000000ULL + uint64_t(std::max(0, c.options.physical_batch - 4)) * (32u << 20)));
    require(c.metrics.estimated_process_bytes <= c.options.process_ceiling_bytes, "Flint dense/KV/bank/scratch admission exceeds memory budget");
    const std::string manifest_path = (fs::path(directory) / "flint-session-manifest.json").string();
    const int fd = open(manifest_path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0600);
    require(fd >= 0, "Cannot create private Flint manifest");
    c.manifest = manifest_path;
    const std::string text = json{{"schema_version", 1}, {"experts_kind", "omni_moe_gguf"}, {"layout", "gguf_tensor_major"},
        {"model", {{"expert_count", experts}, {"expert_used_count", topk}}}, {"entries", entries}}.dump();
    size_t done = 0;
    while (done < text.size()) {
        const ssize_t n = write(fd, text.data() + done, text.size() - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); throw std::runtime_error("Private Flint manifest write failed"); }
        done += size_t(n);
    }
    require(close(fd) == 0, "Flint manifest close failed");
}

flf_context * flf_open(int fd, const char * directory, flf_options o, flf_abort abort, flf_progress progress, void * user) {
    error_text.clear();
    bool expected = false;
    if (!leased.compare_exchange_strong(expected, true)) { error_text = "One Flint context is already loaded"; return nullptr; }
    std::unique_ptr<flf_context> c;
    try {
        c = std::make_unique<flf_context>();
        require(o.context_tokens >= 32 && o.context_tokens <= 65536 && o.evaluation_batch >= 1 && o.evaluation_batch <= 2048 &&
            o.physical_batch >= 1 && o.physical_batch <= 16 && o.physical_batch <= o.evaluation_batch && o.cpu_threads >= 1 &&
            o.cpu_threads <= 8 && o.process_ceiling_bytes > 0 && o.process_ceiling_bytes <= 5500000000ULL && !(o.metal && o.f32_kv_control),
            "Invalid bounded Flint configuration");
        c->options = o; c->abort = abort; c->progress = progress; c->user = user;
        c->source = fcntl(fd, F_DUPFD_CLOEXEC, 0);
        require(c->source >= 0 && fstat(c->source, &c->identity) == 0 && S_ISREG(c->identity.st_mode) && c->identity.st_size > 0,
            "Flint requires a verified open regular file");
        const int model_fd = dup(c->source); require(model_fd >= 0, "Cannot duplicate Flint model descriptor");
        c->stream = fdopen(model_fd, "rb");
        if (!c->stream) { close(model_fd); throw std::runtime_error("Cannot retain Flint GGUF stream"); }
        if (progress) progress(0, -1, user);
        prepare_manifest(*c, directory);
        // Current omni knobs are process environment variables. Caller MUST
        // hold the shared app engine lease before this isolated runtime opens.
        for (const char * key : {"LLAMA_MOE_DECODE_WINDOW", "LLAMA_MOE_LOOKAHEAD_PREFETCH", "LLAMA_MOE_PREFETCH_READ",
            "LLAMA_MOE_SKIP_READ", "LLAMA_MOE_DUMMY_READ", "LLAMA_MOE_MATMUL_SPLIT", "LLAMA_MOE_F_NOCACHE",
            "LLAMA_MOE_READ_DELAY_US", "LLAMA_MOE_PARK_WINDOW"})
            require(setenv(key, "0", 1) == 0, "Cannot configure bounded Flint streaming");
        unsetenv("LLAMA_MOE_LAYER_WINDOWS"); unsetenv("LLAMA_MOE_PREWARM"); unsetenv("LLAMA_MOE_SWEEP");
#if !defined(__APPLE__) || TARGET_OS_OSX
        if (!o.metal) setenv("ORNITH_LES_CPU_CONTROL", "1", 1);
#endif
        std::call_once(initialized, [] { llama_backend_init(); });
        require(!o.metal || ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr,
            "Requested Flint GPU backend is unavailable; CPU fallback is not enabled");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = o.metal ? 999 : 0; mp.use_mmap = o.metal; mp.use_mlock = false; mp.use_extra_bufts = false;
        mp.moe_experts_path = c->manifest.c_str(); mp.moe_slot_bank = c->slots;
        mp.progress_callback_user_data = c.get();
        mp.progress_callback = [](float value, void * opaque) {
            auto * c = static_cast<flf_context *>(opaque);
            if (c->progress) c->progress(1, value, c->user);
            return !c->should_abort();
        };
        require(fseeko(c->stream, 0, SEEK_SET) == 0, "Cannot rewind Flint model stream");
        c->model = llama_model_load_from_file_ptr(c->stream, mp); require(c->model, "Flint model load failed");
        c->check(); if (progress) progress(2, -1, user);
        auto cp = llama_context_default_params();
        cp.n_ctx = o.context_tokens; cp.n_batch = o.evaluation_batch; cp.n_ubatch = o.physical_batch; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = o.cpu_threads; cp.swa_full = false;
        cp.type_k = cp.type_v = o.f32_kv_control ? GGML_TYPE_F32 : GGML_TYPE_F16;
        cp.flash_attn_type = o.metal ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.abort_callback_data = c.get(); cp.abort_callback = [](void * p) { return static_cast<flf_context *>(p)->should_abort(); };
        cp.cb_eval_user_data = c.get();
        cp.cb_eval = o.diagnostic_routing ? +[](ggml_tensor * tensor, bool ask, void * opaque) {
            auto & c = *static_cast<flf_context *>(opaque);
            if (!c.trace_routes) return !ask;
            const std::string name = ggml_get_name(tensor), prefix = "ffn_moe_topk-";
            const bool topk = name.rfind(prefix, 0) == 0;
            if (ask) return topk;
            if (!topk) return true;
            try {
                const int layer = std::stoi(name.substr(prefix.size()));
                if (layer < 0 || layer >= 49 || tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 6 ||
                    tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->ne[1] < 1 || tensor->ne[1] > 2048 ||
                    c.routes.size() + tensor->ne[1] > 32768) return false;
                for (int row = 0; row < tensor->ne[1]; ++row) {
                    // The final FFN is gathered to requested output rows. Its
                    // row 0 represents the last input position, not position 0.
                    if (layer == c.layers - 1 && (!c.trace_output_requested || tensor->ne[1] != 1 || c.trace_layer_rows[layer] != 0)) return false;
                    const int32_t position = layer == c.layers - 1 ? c.trace_final_position :
                        int32_t(c.metrics.cached_tokens + c.trace_layer_rows[layer] + row);
                    flf_context::route entry{position, layer, {}};
                    ggml_backend_tensor_get(tensor, entry.experts.data(), size_t(row) * tensor->nb[1], 6 * sizeof(int32_t));
                    for (int id : entry.experts) if (id < 0 || id >= 256) return false;
                    c.routes.push_back(entry);
                }
                c.trace_layer_rows[layer] += uint32_t(tensor->ne[1]);
                return !c.should_abort();
            } catch (...) { return false; }
        } : nullptr;
        c->context = llama_init_from_model(c->model, cp); require(c->context, "Flint KV/graph allocation failed");
        c->metrics.flash_attention_active = c->context->get_cparams().flash_attn;
        c->check(); if (progress) progress(3, 1, user);
        return c.release();
    } catch (const std::exception & e) {
        if (!c) leased.store(false);
        error_text = e.what(); return nullptr;
    }
}
void flf_close(flf_context * c) { delete c; }
void flf_cancel(flf_context * c) { if (c) c->cancelled.store(true); }
int flf_reset(flf_context * c) {
    if (!c) { error_text = "Flint context is null"; return -1; }
    llama_memory_clear(llama_get_memory(c->context), false);
    c->metrics.cached_tokens = 0; c->cancelled.store(false); c->failed = false; c->logits_valid = false;
    c->routes.clear(); c->trace_layer_rows.fill(0);
    if (c->sampler) llama_sampler_reset(c->sampler);
    return 0;
}
int32_t flf_vocab_size(const flf_context * c) { return c ? llama_vocab_n_tokens(llama_model_get_vocab(c->model)) : 0; }
int flf_tokenize(flf_context * c, const char * text, int32_t bytes, int32_t * tokens, int32_t capacity) {
    if (!c || !text || bytes < 0 || bytes > (2 << 20) || capacity < 0 || (capacity && !tokens)) { error_text = "Invalid Flint tokenizer buffer"; return 0; }
    return llama_tokenize(llama_model_get_vocab(c->model), text, bytes, tokens, capacity, false, true);
}
int flf_token_piece(flf_context * c, int32_t token, char * text, int32_t capacity) {
    if (!c || token < 0 || token >= flf_vocab_size(c) || capacity < 0 || (capacity && !text)) { error_text = "Invalid Flint token"; return 0; }
    return llama_token_to_piece(llama_model_get_vocab(c->model), token, text, capacity, 0, true);
}
bool flf_is_eog(flf_context * c, int32_t token) { return c && token >= 0 && token < flf_vocab_size(c) && llama_vocab_is_eog(llama_model_get_vocab(c->model), token); }
static int evaluate(flf_context * c, const int32_t * tokens, int32_t count, bool final_row, float * logits, int32_t capacity) {
    error_text.clear();
    try {
        require(c && tokens && count > 0 && count <= c->options.evaluation_batch && !c->failed, "Invalid Flint evaluation or failed session; reset required");
        require(c->metrics.cached_tokens + uint64_t(count) <= uint64_t(c->options.context_tokens), "Flint context capacity exceeded");
        require(!logits || capacity >= flf_vocab_size(c), "Insufficient Flint logit buffer");
        for (int i = 0; i < count; ++i) require(tokens[i] >= 0 && tokens[i] < flf_vocab_size(c), "Invalid Flint input token");
        c->check();
        c->logits_valid = false;
        c->trace_layer_rows.fill(0);
        c->trace_final_position = int32_t(c->metrics.cached_tokens + count - 1);
        c->trace_output_requested = final_row;
        llama_batch batch = llama_batch_init(count, 0, 1); batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = c->metrics.cached_tokens + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = final_row && i == count - 1;
        }
        const auto start = std::chrono::steady_clock::now();
        const int status = llama_decode(c->context, batch); llama_batch_free(batch);
        llama_synchronize(c->context);
        c->metrics.forward_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (status || c->should_abort()) {
            c->failed = true; llama_memory_clear(llama_get_memory(c->context), false); c->metrics.cached_tokens = 0;
            throw std::runtime_error("Flint evaluation cancelled/failed; partial KV cleared, reset required");
        }
        c->metrics.cached_tokens += count; c->metrics.evaluated_tokens += count;
        if (final_row) {
            const auto * result = llama_get_logits_ith(c->context, -1);
            require(result, "Flint returned no requested logits");
            if (logits) std::memcpy(logits, result, flf_vocab_size(c) * sizeof(float));
            c->logits_valid = true;
        }
        else c->metrics.state_only_tokens += count;
        c->check(); return 0;
    } catch (const std::exception & e) { error_text = e.what(); return -1; }
}
int flf_evaluate(flf_context * c, const int32_t * tokens, int32_t count, float * logits, int32_t capacity) {
    return evaluate(c, tokens, count, logits != nullptr, logits, capacity);
}
int flf_evaluate_for_sampling(flf_context * c, const int32_t * tokens, int32_t count) {
    return evaluate(c, tokens, count, true, nullptr, 0);
}
int flf_copy_logits(flf_context * c, float * logits, int32_t capacity) {
    error_text.clear();
    try {
        require(c && !c->failed && c->logits_valid && logits && capacity >= flf_vocab_size(c), "No current Flint logits or invalid diagnostic buffer");
        c->check();
        const auto * row = llama_get_logits_ith(c->context, -1);
        require(row, "Flint logit row missing");
        std::memcpy(logits, row, flf_vocab_size(c) * sizeof(float)); return flf_vocab_size(c);
    } catch (const std::exception & e) { error_text = e.what(); return -1; }
}
int flf_configure_sampling(flf_context * c, float temperature, float top_p, int32_t top_k, uint32_t seed) {
    error_text.clear();
    try {
        require(c && !c->failed && std::isfinite(temperature) && temperature >= 0 && temperature <= 2 &&
            std::isfinite(top_p) && top_p > 0 && top_p <= 1 && top_k >= 0 && top_k <= flf_vocab_size(c), "Invalid Flint sampling configuration");
        c->check();
        auto params = llama_sampler_chain_default_params(); params.no_perf = true;
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> chain(llama_sampler_chain_init(params), llama_sampler_free);
        require(bool(chain), "Cannot allocate Flint sampler");
        if (temperature == 0) llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        else {
            llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(top_k));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(top_p, 1));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
        }
        if (c->sampler) llama_sampler_free(c->sampler);
        c->sampler = chain.release(); c->candidates.resize(flf_vocab_size(c)); return 0;
    } catch (const std::exception & e) { error_text = e.what(); return -1; }
}
int32_t flf_sample(flf_context * c) {
    error_text.clear();
    try {
        require(c && !c->failed && c->sampler && c->logits_valid, "Configure sampling and evaluate final Flint logits first");
        c->check();
        const auto * logits = llama_get_logits_ith(c->context, -1);
        require(logits, "Flint logit row missing");
        // Same GGML sampling chain as llama_sampler_sample, with one reusable
        // candidate buffer instead of allocating a vocabulary vector per step.
        for (int32_t i = 0; i < flf_vocab_size(c); ++i) {
            require(std::isfinite(logits[i]), "Non-finite Flint logits; sampling stopped");
            c->candidates[i] = {i, logits[i], 0};
        }
        llama_token_data_array values{c->candidates.data(), c->candidates.size(), -1, false};
        llama_sampler_apply(c->sampler, &values);
        require(values.selected >= 0 && uint64_t(values.selected) < values.size, "Flint sampler returned no selection");
        const auto token = values.data[values.selected].id;
        llama_sampler_accept(c->sampler, token); return token;
    } catch (const std::exception & e) { error_text = e.what(); return -1; }
}
int flf_trace_routes(flf_context * c, bool enabled) {
    error_text.clear();
    try {
        require(c && !c->failed && (!enabled || c->options.diagnostic_routing), "Flint routing callback was not enabled at load"); c->check();
        c->trace_routes = enabled; c->routes.clear(); c->trace_layer_rows.fill(0); return 0;
    } catch (const std::exception & e) { error_text = e.what(); return -1; }
}
int flf_routing_json(flf_context * c, char * output, int32_t capacity) {
    error_text.clear();
    try {
        require(c && !c->failed && c->trace_routes && capacity >= 0 && (!capacity || output), "Enable bounded Flint routing diagnostics first");
        c->check(); auto entries = json::array();
        for (const auto & row : c->routes) entries.push_back({{"position", row.position}, {"layer", row.layer}, {"experts", row.experts}});
        const auto text = entries.dump();
        require(text.size() <= (4u << 20), "Flint routing diagnostic exceeds output budget");
        if (size_t(capacity) < text.size()) return -int(text.size());
        std::memcpy(output, text.data(), text.size()); return int(text.size());
    } catch (const std::exception & e) { error_text = e.what(); return 0; }
}
flf_metrics flf_snapshot(const flf_context * c) {
    if (!c) return {};
    auto out = c->metrics;
    if (auto * slots = c->context->flash_local_slots()) { auto s = slots->counters_snapshot(); out.io_bytes = s.all_bytes; out.io_calls = s.all_calls; }
    return out;
}
const char * flf_last_error() { return error_text.c_str(); }
const char * flf_revision() { return "96bbf617+flint-a37cc461-backport-2"; }
const char * flf_library_path() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void *>(&flf_revision), &info) || !info.dli_fname) {
        error_text = "Cannot identify the loaded Flint framework"; return nullptr;
    }
    return info.dli_fname;
}
