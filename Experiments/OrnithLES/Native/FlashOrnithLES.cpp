#include "FlashOrnithLES.h"
#include "LoaderDiagnostics.h"
#include "RoutingTrace.h"
#include "ProjectionProbe.h"
#include "../../../Sources/CLlamaTokenizer/CheckpointObserver.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama-moe-slot.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <TargetConditionals.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <map>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <vector>
using json = nlohmann::json;
namespace fs = std::filesystem;
extern "C" void flash_omni_metal_snapshot(uint64_t *, uint64_t *, uint64_t *, uint64_t *);
extern "C" void flash_omni_memory_snapshot(uint64_t *, uint64_t *, uint64_t *, uint64_t *, bool *);
static std::atomic<uint64_t> live_contexts{0};
static thread_local std::string last_error;
static std::atomic_flag owner = ATOMIC_FLAG_INIT;
// Swift actor calls are serial but may resume on different OS threads.
// This is an instance lease, not a thread-affine mutex held across calls.
struct context_lease {
    const bool acquired = !owner.test_and_set(std::memory_order_acquire);
    context_lease() = default;
    context_lease(const context_lease &) = delete;
    context_lease & operator=(const context_lease &) = delete;
    ~context_lease() { if (acquired) owner.clear(std::memory_order_release); }
};
static std::once_flag initialized;
static flash_local_loader_diagnostics loader_diagnostics;
struct loader_log_scope {
    loader_log_scope() { loader_diagnostics.begin(); }
    ~loader_log_scope() { loader_diagnostics.end(); }
};
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
static std::string fd_path(int fd) { return "/dev/fd/" + std::to_string(fd); }
static uint64_t file_size(int fd) {
    struct stat s{};
    require(fstat(fd, &s) == 0 && S_ISREG(s.st_mode) && s.st_size > 0, "Invalid verified descriptor");
    return (uint64_t)s.st_size;
}
static void check_abort(flo_abort abort, void * user) {
    if (abort && abort(user)) throw std::runtime_error("Cancelled or memory admission revoked");
}
struct fd_owner {
    int value = -1;
    ~fd_owner() { if (value >= 0) close(value); }
};
struct gguf_owner {
    ggml_context * tensors = nullptr;
    gguf_context * file = nullptr;
    ~gguf_owner() { if (file) gguf_free(file); if (tensors) ggml_free(tensors); }
};
static void transfer(int fd, void * buffer, size_t size, uint64_t offset, bool writing) {
    size_t done = 0;
    while (done < size) {
        ssize_t n = writing ? pwrite(fd, (char *)buffer + done, size - done, offset + done)
                            : pread(fd, (char *)buffer + done, size - done, offset + done);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "Incomplete local model I/O"); done += (size_t)n;
    }
}
static uint32_t meta_u32(gguf_context * file, const char * key) {
    const auto id = gguf_find_key(file, key);
    require(id >= 0 && gguf_get_kv_type(file, id) == GGUF_TYPE_UINT32, "Missing integer model metadata");
    return gguf_get_val_u32(file, id);
}

int flo_prepare(int source, const char * directory, flo_abort abort, void * user) {
    try {
        last_error.clear(); const auto source_bytes = file_size(source);
        require(directory && fs::is_directory(directory) && fs::is_empty(directory), "Preparation needs an empty private directory");
        check_abort(abort, user);
        const int header_fd = dup(source); require(header_fd >= 0, "Cannot duplicate GGUF descriptor");
        FILE * raw_header = fdopen(header_fd, "rb");
        if (!raw_header) { close(header_fd); throw std::runtime_error("Cannot open GGUF stream"); }
        std::unique_ptr<FILE, decltype(&fclose)> header_file(raw_header, &fclose);
        require(fseek(raw_header, 0, SEEK_SET) == 0, "Cannot rewind verified GGUF stream");
        gguf_owner input, dense;
        input.file = gguf_init_from_file_ptr(raw_header, {true, &input.tensors});
        require(input.file && input.tensors, "Invalid GGUF metadata");
        const auto arch = gguf_find_key(input.file, "general.architecture");
        require(arch >= 0 && gguf_get_kv_type(input.file, arch) == GGUF_TYPE_STRING &&
            std::string(gguf_get_val_str(input.file, arch)) == "qwen35moe", "LES currently admits qwen35moe only");
        const auto count = meta_u32(input.file, "qwen35moe.expert_count");
        const auto used = meta_u32(input.file, "qwen35moe.expert_used_count");
        require(count > 0 && count <= 256 && used > 0 && used <= count, "Invalid expert geometry");
        dense.file = gguf_init_empty(); require(dense.file, "Dense metadata allocation failed");
        gguf_set_kv(dense.file, input.file);
        std::vector<int64_t> kept;
        json entries = json::array();
        std::regex routed("^blk\\.([0-9]+)\\.(ffn_(gate|up|down|gate_up)_exps)\\.weight$");
        for (int64_t i = 0; i < gguf_get_n_tensors(input.file); ++i) {
            std::string name = gguf_get_tensor_name(input.file, i); std::smatch match;
            const auto * tensor = ggml_get_tensor(input.tensors, name.c_str());
            require(tensor, "Missing tensor descriptor");
            if (!std::regex_match(name, match, routed)) {
                gguf_add_tensor(dense.file, tensor); kept.push_back(i); continue;
            }
            const size_t bytes = ggml_nbytes(tensor);
            require(tensor->ne[2] == count && tensor->ne[3] == 1 && bytes % count == 0, "Invalid expert tensor layout");
            const uint64_t offset = gguf_get_data_offset(input.file) + gguf_get_tensor_offset(input.file, i);
            require(offset <= source_bytes && bytes <= source_bytes - offset, "Expert exceeds verified source");
            entries.push_back({{"layer", std::stoi(match[1])}, {"tensor_family", match[2].str()},
                {"tensor_name", name}, {"original_gguf_tensor_name", name},
                {"quant_type", ggml_type_name(tensor->type)}, {"exact_byte_length", bytes},
                {"bytes_per_expert", bytes / count}, {"expert_stride", bytes / count},
                {"repacked_offset", offset}, {"repacked_file", "verified-descriptor-required"}});
        }
        require(!entries.empty(), "No routed experts");
        const fs::path output = fs::path(directory) / "dense.gguf";
        require(!fs::exists(output), "Dense output already exists");
        require(gguf_write_to_file(dense.file, output.c_str(), true), "Dense header write failed");
        fd_owner target{open(output.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC)};
        require(target.value >= 0, "Cannot open dense output");
        std::vector<uint8_t> buffer(4 << 20), readback(buffer.size());
        uint64_t end = gguf_get_meta_size(dense.file);
        for (size_t index = 0; index < kept.size(); ++index) {
            const auto i = kept[index];
            uint64_t remaining = gguf_get_tensor_size(input.file, i);
            uint64_t from = gguf_get_data_offset(input.file) + gguf_get_tensor_offset(input.file, i);
            uint64_t to = gguf_get_meta_size(dense.file) + gguf_get_tensor_offset(dense.file, index);
            require(from <= source_bytes && remaining <= source_bytes - from, "Dense tensor exceeds source");
            end = std::max(end, to + remaining);
            while (remaining) {
                check_abort(abort, user); const size_t n = std::min<uint64_t>(remaining, buffer.size());
                transfer(source, buffer.data(), n, from, false);
                transfer(target.value, buffer.data(), n, to, true);
                transfer(target.value, readback.data(), n, to, false);
                require(std::equal(buffer.begin(), buffer.begin() + n, readback.begin()), "Dense readback mismatch");
                remaining -= n; from += n; to += n;
            }
        }
        require(ftruncate(target.value, end) == 0 && fsync(target.value) == 0, "Dense flush failed");
        json manifest{{"schema_version", 1}, {"experts_kind", "omni_moe_gguf"},
            {"layout", "gguf_tensor_major"}, {"source_file_bytes", source_bytes},
            {"model", {{"arch", "qwen35moe"}, {"expert_count", count}, {"expert_used_count", used}}},
            {"entries", entries}};
        std::ofstream out(fs::path(directory) / "manifest-template.json", std::ios::binary);
        out << manifest.dump(2) << '\n'; out.close(); require(bool(out), "Manifest write failed");
        return 0;
    } catch (const std::exception & e) { last_error = e.what(); return -1; }
}

struct flo_draft {
    fd_owner source;
    FILE * file = nullptr;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    std::vector<int32_t> layers;
    std::vector<float> features;
    bool mrope = false;
    ~flo_draft() {
        if (context) { llama_synchronize(context); llama_free(context); }
        if (model) llama_model_free(model);
        if (file) fclose(file);
    }
};
struct flo_context {
    context_lease exclusive; // first member: released after all other members
    std::unique_ptr<flo_draft> draft;
    std::vector<uint8_t> prefix_state;
    int prefix_position = 0;
    fd_owner source;
    fs::path manifest;
    llama_model * model = nullptr;
    FILE * model_file = nullptr;
    llama_context * context = nullptr;
    flo_abort abort = nullptr; void * user = nullptr;
    int vocabulary = 0, batch = 0, position = 0, capacity = 0;
    int maximum_physical_batch = 0;
    uint64_t skipped_intermediate_outputs = 0;
    uint64_t target_tokens = 0, draft_blocks = 0, draft_proposals = 0, draft_accepted = 0, draft_replays = 0, feature_peak = 0;
    uint64_t additional_reserve = 0;
    bool experimental_dflash = false, metal = false;
    bool poisoned = false, routing_failed = false;
    flo_checkpoint checkpoint = nullptr;
    void * checkpoint_user = nullptr;
    flash_local_projection_probe projection_probe;
    flash_local_routing_trace routing;
    explicit flo_context() { live_contexts.fetch_add(1, std::memory_order_relaxed); }
    ~flo_context() {
        draft.reset(); // shared embedding/head must outlive the draft context
        // Complete queued GPU work before releasing recurrent state and model
        // buffers, including failed/cancelled forward paths.
        if (context) llama_synchronize(context);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
        if (model_file) fclose(model_file);
        if (!manifest.empty()) { std::error_code ignored; fs::remove(manifest, ignored); }
        live_contexts.fetch_sub(1, std::memory_order_relaxed);
    }
};
static void environment(const char * name, int value) {
    require(setenv(name, std::to_string(value).c_str(), 1) == 0, "Cannot configure isolated LES runtime");
}
flo_context * flo_open(int source, const char * directory, flo_options o, flo_abort abort, void * user) {
    try {
        last_error.clear(); auto result = std::make_unique<flo_context>();
        require(result->exclusive.acquired, "Only one LES context may own the engine");
        loader_log_scope diagnostic_scope;
        result->source.value = dup(source); require(result->source.value >= 0, "Cannot retain verified descriptor");
        const auto size = file_size(result->source.value); check_abort(abort, user);
        require(o.context_tokens >= 32 && o.context_tokens <= 65536 && o.memory_ceiling_bytes > 0 &&
            o.memory_ceiling_bytes <= 5500000000ULL, "Invalid context or process ceiling");
        require(o.additional_reserve_bytes <= 1500000000ULL &&
            (!o.experimental_dflash || (o.context_tokens <= 32768 && o.additional_reserve_bytes >= 800000000ULL &&
                o.prefill_slots == 32)),
            "DFlash benchmark needs explicit additional memory, physical batch four and at most 32K context");
        result->experimental_dflash = o.experimental_dflash;
        result->additional_reserve = o.additional_reserve_bytes; result->metal = o.metal;
        require(o.lookahead_depth >= 0 && o.lookahead_depth <= 16 && o.io_threads >= 1 && o.io_threads <= 8 &&
            o.cpu_threads >= 1 && o.cpu_threads <= 8, "Invalid scheduling options");
        const bool streaming = directory && *directory;
        if (streaming && !o.metal) {
            #if TARGET_OS_OSX
            const char * control_mode = std::getenv("ORNITH_LES_CPU_CONTROL");
            require(control_mode && std::string(control_mode) == "1" && size <= (32 << 20),
                "CPU streaming is restricted to bounded macOS synthetic controls");
            #else
            throw std::runtime_error("The iPhone LES backend requires Metal");
            #endif
        }
        require(streaming || size <= (32 << 20), "Resident control is restricted to tiny synthetic fixtures");
        require(!streaming || (o.prefill_slots >= 8 && o.prefill_slots <= 256 &&
            (o.cache_slots_per_layer == 0 || o.cache_slots_per_layer >= 8) &&
            o.cache_slots_per_layer <= 256), "Invalid expert cache geometry");
        // Disable all diagnostic shortcuts that can produce incorrect outputs.
        for (const char * name : {"LLAMA_MOE_SKIP_READ", "LLAMA_MOE_DUMMY_READ", "LLAMA_MOE_MATMUL_SPLIT",
                "LLAMA_MOE_READ_DELAY_US", "LLAMA_MOE_PARK_WINDOW"}) environment(name, 0);
        unsetenv("LLAMA_MOE_LAYER_WINDOWS"); unsetenv("LLAMA_MOE_PREWARM"); unsetenv("LLAMA_MOE_SWEEP");
        environment("LLAMA_MOE_PREFETCH_READ", o.lookahead_depth);
        environment("LLAMA_MOE_LOOKAHEAD_PREFETCH", o.lookahead_depth > 0);
        environment("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", o.lookahead_depth);
        environment("LLAMA_MOE_DECODE_WINDOW", streaming ? o.cache_slots_per_layer : 0);
        environment("LLAMA_MOE_WINDOW_MAX_ROWS", 1);
        environment("LLAMA_MOE_IO_THREADS", o.io_threads);
        environment("LLAMA_MOE_F_NOCACHE", o.bypass_page_cache);
        std::string model_path = fd_path(result->source.value);
        std::string manifest_path;
        if (streaming) {
            std::ifstream in(fs::path(directory) / "manifest-template.json"); json manifest; in >> manifest;
            require(manifest.at("source_file_bytes").get<uint64_t>() == size, "Prepared source length changed");
            require(manifest.at("layout") == "gguf_tensor_major", "Unsupported local expert layout");
            require(o.lookahead_depth <= manifest.at("model").at("expert_count").get<int64_t>(),
                "LES lookahead exceeds expert count");
            std::map<std::pair<std::string, uint64_t>, uint64_t> groups;
            for (const auto & entry : manifest.at("entries")) {
                const uint64_t bytes = entry.at("bytes_per_expert");
                require(bytes > 0 && bytes <= (16 << 20), "Invalid expert extent");
                groups[{entry.at("tensor_family").get<std::string>(), bytes}]++;
            }
            uint64_t admitted = fs::file_size(fs::path(directory) / "dense.gguf") + (1ULL << 30)
                + (uint64_t)o.context_tokens * 20480 + 64 * 1024 * 1024;
            admitted += o.additional_reserve_bytes;
            admitted += uint64_t(std::max(0, std::min(16, o.prefill_slots / 8) - 4)) * (32ULL << 20);
            for (const auto & group : groups)
                admitted += group.first.second * (o.prefill_slots + (o.cache_slots_per_layer > 0 ? group.second *
                    (o.cache_slots_per_layer + o.lookahead_depth) : 0));
            require(admitted <= o.memory_ceiling_bytes, "LES dense/cache/KV/scratch admission exceeds process ceiling");
            for (auto & entry : manifest.at("entries")) entry["repacked_file"] = model_path;
            result->manifest = fs::path(directory) / "session-manifest.json";
            std::ofstream out(result->manifest); out << manifest.dump(); out.close(); require(bool(out), "Session manifest write failed");
            manifest_path = result->manifest.string(); model_path = (fs::path(directory) / "dense.gguf").string();
        }
        std::call_once(initialized, [] {
            llama_log_set([](ggml_log_level level, const char * text, void *) {
                if (text) fputs(text, stderr);
                loader_diagnostics.record(level, text);
            }, nullptr);
            llama_backend_init();
        });
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = o.metal ? 999 : 0; mp.use_mmap = o.metal; mp.use_mlock = false; mp.use_extra_bufts = false;
        if (streaming) { mp.moe_experts_path = manifest_path.c_str(); mp.moe_slot_bank = o.prefill_slots; }
        mp.progress_callback = [](float, void * data) {
            auto * c = static_cast<flo_context *>(data); return !c->abort || !c->abort(c->user);
        };
        result->abort = abort; result->user = user; mp.progress_callback_user_data = result.get();
        const int model_fd = streaming ? open(model_path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC) : dup(result->source.value);
        require(model_fd >= 0, "Cannot open LES model stream");
        result->model_file = fdopen(model_fd, "rb");
        if (!result->model_file) { close(model_fd); throw std::runtime_error("Cannot retain LES model stream"); }
        require(fseek(result->model_file, 0, SEEK_SET) == 0, "Cannot rewind model stream");
        result->model = llama_model_load_from_file_ptr(result->model_file, mp);
        if (!result->model) throw std::runtime_error(loader_diagnostics.describe("LES model loading failed"));
        check_abort(abort, user);
        auto cp = llama_context_default_params();
        // Extracted target-layer buffers reserve n_batch rows, not just the
        // current microbatch. Bound them in the DFlash-only context.
        cp.n_ctx = o.context_tokens; cp.n_batch = o.experimental_dflash ? 16 : 2048;
        cp.n_ubatch = streaming ? std::max(1, std::min(16, o.prefill_slots / 8)) : 1;
        cp.n_threads = cp.n_threads_batch = o.cpu_threads; cp.n_seq_max = 1; cp.n_rs_seq = 0;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = o.flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf = false; cp.abort_callback = abort; cp.abort_callback_data = user;
        result->checkpoint = o.checkpoint; result->checkpoint_user = o.checkpoint_user;
        if (o.trace_routing || o.checkpoint) {
            cp.cb_eval_user_data = result.get();
            cp.cb_eval = [](ggml_tensor * tensor, bool ask, void * data) {
                auto * c = static_cast<flo_context *>(data);
                const bool routed = std::string(ggml_get_name(tensor)).rfind("ffn_moe_topk-", 0) == 0;
                const int32_t final_layer = int32_t(c->model->hparams.n_layer) - int32_t(c->model->hparams.nextn_predict_layers) - 1;
                const auto key = c->checkpoint ? flash_local_checkpoint::identify(ggml_get_name(tensor), final_layer) : flash_local_checkpoint::key{};
                const bool checkpoint = key.stage != 0;
                const bool selected = routed || checkpoint;
                if (ask) return selected;
                if (!selected) return true;
                if (checkpoint) {
                    if (c->position == 0 && key.stage == 3 && !c->projection_probe.samples) {
                        // Unsupported layouts remain explicitly unmeasured.
                        (void)c->projection_probe.run(tensor);
                    }
                    const bool valid = flash_local_checkpoint::emit(tensor, c->checkpoint, c->checkpoint_user, nullptr, final_layer);
                    c->routing_failed = c->routing_failed || !valid;
                    return valid;
                }
                const bool valid = tensor->type == GGML_TYPE_I32 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
                    c->routing.append(ggml_get_name(tensor), c->position, tensor->ne[0], tensor->ne[1],
                        tensor->nb[1], c->model->hparams.n_expert,
                        [tensor](int32_t * row, size_t offset, size_t bytes) {
                            ggml_backend_tensor_get(tensor, row, offset, bytes);
                        });
                c->routing_failed = c->routing_failed || !valid;
                return valid;
            };
        }
        result->context = llama_init_from_model(result->model, cp);
        if (!result->context) throw std::runtime_error(loader_diagnostics.describe("LES context allocation failed"));
        result->vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(result->model));
        result->batch = (int)cp.n_ubatch; result->capacity = o.context_tokens;
        check_abort(abort, user); return result.release();
    } catch (const std::exception & e) { last_error = e.what(); return nullptr; }
}
void flo_close(flo_context * c) { delete c; }
int32_t flo_vocabulary_size(const flo_context * c) { return c ? c->vocabulary : 0; }
static void draft_process(flo_context *, int position, int count);
int flo_decode(flo_context * c, const int32_t * tokens, int32_t count, float * output, int32_t capacity) {
    try {
        require(c && !c->poisoned && tokens && output && count > 0 && count <= 2048 && capacity == c->vocabulary &&
            count <= c->capacity - c->position, "Invalid LES decode request");
        for (int i = 0; i < count; ++i) require(tokens[i] >= 0 && tokens[i] < c->vocabulary, "Token outside vocabulary");
        for (int first = 0; first < count; first += c->batch) {
            check_abort(c->abort, c->user); const int n = std::min(c->batch, count - first);
            auto batch = llama_batch_get_one(const_cast<int32_t *>(tokens + first), n);
            // All recurrent/KV work and real expert choices are still evaluated.
            // Only the final microbatch's logits escape this API. The pinned
            // Qwen graph selects output rows after all transformer layers.
            int8_t no_output[16]{};
            const bool intermediate = first + n < count;
            if (intermediate) batch.logits = no_output;
            require(llama_decode(c->context, batch) == 0, "LES forward failed");
            c->target_tokens += n;
            if (c->draft) draft_process(c, c->position, n);
            if (intermediate) ++c->skipped_intermediate_outputs;
            c->maximum_physical_batch = std::max(c->maximum_physical_batch, n);
            // ggml's evaluation callback can stop a split without returning an
            // error status. Never export logits from an invalid trace.
            require(!c->routing_failed, "LES routing observation failed"); c->position += n;
        }
        const float * logits = llama_get_logits_ith(c->context, -1); require(logits, "LES logits unavailable");
        for (int i = 0; i < capacity; ++i) require(std::isfinite(logits[i]), "Non-finite LES logit");
        std::copy(logits, logits + capacity, output); check_abort(c->abort, c->user); return 0;
    } catch (const std::exception & e) { if (c) c->poisoned = true; last_error = e.what(); return -1; }
}
void flo_reset(flo_context * c) {
    if (c) { std::vector<uint8_t>().swap(c->prefix_state); c->prefix_position = 0; }
    if (c && !c->poisoned) {
        llama_memory_clear(llama_get_memory(c->context), true); c->position = 0;
        if (c->draft) llama_memory_clear(llama_get_memory(c->draft->context), true);
        c->routing = {};
    }
}
uint64_t flo_capture_prefix(flo_context * c) {
    try {
        require(c && !c->poisoned && c->position > 0, "Invalid prefix checkpoint");
        check_abort(c->abort, c->user);
        c->prefix_state.clear(); c->prefix_position = 0;
        const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t size = llama_state_seq_get_size_ext(c->context, 0, flags);
        require(size > 0 && size <= (128ULL << 20), "Recurrent checkpoint exceeds 128 MiB");
        c->prefix_state.resize(size);
        require(llama_state_seq_get_data_ext(c->context, c->prefix_state.data(), size, 0, flags) == size,
            "Incomplete recurrent checkpoint");
        c->prefix_position = c->position;
        return size;
    } catch (const std::exception & e) {
        if (c) { std::vector<uint8_t>().swap(c->prefix_state); c->prefix_position = 0; }
        last_error = e.what(); return 0;
    }
}
int flo_restore_prefix(flo_context * c) {
    try {
        require(c && !c->poisoned && c->prefix_position > 0 && !c->prefix_state.empty(), "No reusable prefix checkpoint");
        check_abort(c->abort, c->user);
        // The pinned hybrid implementation restores only recurrent state for
        // PARTIAL_ONLY. Restore that first, then discard attention suffix cells;
        // recurrent seq_rm now sees no state beyond the checkpoint position.
        require(llama_state_seq_set_data_ext(c->context, c->prefix_state.data(), c->prefix_state.size(),
            0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == c->prefix_state.size(), "Prefix state restore failed");
        require(llama_memory_seq_rm(llama_get_memory(c->context), 0, c->prefix_position, -1), "Prefix KV suffix trim failed");
        c->position = c->prefix_position;
        return 0;
    } catch (const std::exception & e) { if (c) c->poisoned = true; last_error = e.what(); return -1; }
}
flo_metrics flo_snapshot(const flo_context * c) {
    flo_metrics out{};
    if (c) {
        out.routing_digest = c->routing.sequential; out.routing_decisions = c->routing.decisions;
        out.canonical_routing_digest = c->routing.canonical;
        out.canonical_expert_set_digest = c->routing.expert_sets;
        out.configured_physical_batch_tokens = c->batch;
        out.maximum_physical_batch_tokens = c->maximum_physical_batch;
        out.skipped_intermediate_output_batches = c->skipped_intermediate_outputs;
        out.target_tokens_evaluated = c->target_tokens; out.dflash_blocks = c->draft_blocks;
        out.dflash_proposals = c->draft_proposals; out.dflash_accepted = c->draft_accepted;
        out.dflash_replays = c->draft_replays; out.dflash_feature_peak_bytes = c->feature_peak;
        out.flash_attention_active = c->context && c->context->get_cparams().flash_attn;
        const auto & p = c->projection_probe;
        out.probe_samples = p.samples; out.probe_columns = p.columns;
        out.probe_weight_type = p.weight_type; out.probe_cpu_input_type = p.cpu_input_type;
        out.probe_backend_max = p.backend_max; out.probe_cpu_max = p.cpu_max;
        out.probe_backend_squared = p.backend_squared; out.probe_cpu_squared = p.cpu_squared;
        out.probe_reference_squared = p.reference_squared;
    }
    if (c && c->context->flash_local_slots()) {
        const auto s = c->context->flash_local_slots()->counters_snapshot();
        out.io_bytes = s.all_bytes; out.io_calls = s.all_calls; out.misses = s.miss; out.selected_experts = s.unique;
        out.predicted = s.predicted; out.prediction_hits = s.prediction_hits; out.prefetch_bytes = s.prefetch_bytes;
        out.process_ns = s.process; out.install_ns = s.install;
        out.demand_hits = s.demand_hits; out.demand_misses = s.demand_misses;
    }
    flash_omni_metal_snapshot(&out.dispatches, &out.command_buffers, &out.gpu_ns, &out.timestamped_command_buffers);
    out.live_contexts = live_contexts.load(std::memory_order_relaxed);
    flash_omni_memory_snapshot(&out.live_metal_buffers, &out.owned_metal_bytes, &out.mapped_metal_bytes,
        &out.device_allocated_bytes, &out.metal_device_observed);
    return out;
}
const char * flo_last_error() { return last_error.c_str(); }
const char * flo_revision() { return "96bbf61710d44421ea54e63f5194f9b17c20d877+flash-local-experiments-v7"; }

#include "DFlashBenchmark.inc"
