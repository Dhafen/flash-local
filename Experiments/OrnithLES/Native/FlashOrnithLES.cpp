#include "FlashOrnithLES.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-moe-slot.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
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
extern "C" void flash_omni_metal_snapshot(uint64_t *, uint64_t *, uint64_t *);
static thread_local std::string last_error;
static std::mutex owner;
static std::once_flag initialized;
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
        gguf_owner input, dense;
        input.file = gguf_init_from_file(fd_path(source).c_str(), {true, &input.tensors});
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

struct flo_context {
    std::unique_lock<std::mutex> exclusive;
    fd_owner source;
    fs::path manifest;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    flo_abort abort = nullptr; void * user = nullptr;
    int vocabulary = 0, batch = 0, position = 0, capacity = 0;
    bool poisoned = false;
    uint64_t routing_digest = 14695981039346656037ULL, routing_decisions = 0;
    explicit flo_context() : exclusive(owner, std::try_to_lock) {}
    ~flo_context() {
        if (context) llama_free(context);
        if (model) llama_model_free(model);
        if (!manifest.empty()) { std::error_code ignored; fs::remove(manifest, ignored); }
    }
};
static void environment(const char * name, int value) {
    require(setenv(name, std::to_string(value).c_str(), 1) == 0, "Cannot configure isolated LES runtime");
}
flo_context * flo_open(int source, const char * directory, flo_options o, flo_abort abort, void * user) {
    try {
        last_error.clear(); auto result = std::make_unique<flo_context>();
        require(result->exclusive.owns_lock(), "Only one LES context may own the engine");
        result->source.value = dup(source); require(result->source.value >= 0, "Cannot retain verified descriptor");
        const auto size = file_size(result->source.value); check_abort(abort, user);
        require(o.context_tokens >= 32 && o.context_tokens <= 65536 && o.memory_ceiling_bytes > 0 &&
            o.memory_ceiling_bytes <= 5500000000ULL, "Invalid context or process ceiling");
        require(o.lookahead_depth >= 0 && o.lookahead_depth <= 16 && o.io_threads >= 1 && o.io_threads <= 8 &&
            o.cpu_threads >= 1 && o.cpu_threads <= 8, "Invalid scheduling options");
        const bool streaming = directory && *directory;
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
            std::map<std::pair<std::string, uint64_t>, uint64_t> groups;
            for (const auto & entry : manifest.at("entries")) {
                const uint64_t bytes = entry.at("bytes_per_expert");
                require(bytes > 0 && bytes <= (16 << 20), "Invalid expert extent");
                groups[{entry.at("tensor_family").get<std::string>(), bytes}]++;
            }
            uint64_t admitted = fs::file_size(fs::path(directory) / "dense.gguf") + (1ULL << 30)
                + (uint64_t)o.context_tokens * 20480 + 64 * 1024 * 1024;
            for (const auto & group : groups)
                admitted += group.first.second * (o.prefill_slots + (o.cache_slots_per_layer > 0 ? group.second *
                    (o.cache_slots_per_layer + o.lookahead_depth) : 0));
            require(admitted <= o.memory_ceiling_bytes, "LES dense/cache/KV/scratch admission exceeds process ceiling");
            for (auto & entry : manifest.at("entries")) entry["repacked_file"] = model_path;
            result->manifest = fs::path(directory) / "session-manifest.json";
            std::ofstream out(result->manifest); out << manifest.dump(); out.close(); require(bool(out), "Session manifest write failed");
            manifest_path = result->manifest.string(); model_path = (fs::path(directory) / "dense.gguf").string();
        }
        std::call_once(initialized, [] { llama_backend_init(); });
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = o.metal ? 999 : 0; mp.use_mmap = true; mp.use_mlock = false; mp.use_extra_bufts = false;
        if (streaming) { mp.moe_experts_path = manifest_path.c_str(); mp.moe_slot_bank = o.prefill_slots; }
        mp.progress_callback = [](float, void * data) {
            auto * c = static_cast<flo_context *>(data); return !c->abort || !c->abort(c->user);
        };
        result->abort = abort; result->user = user; mp.progress_callback_user_data = result.get();
        result->model = llama_model_load_from_file(model_path.c_str(), mp);
        require(result->model, "LES model loading failed"); check_abort(abort, user);
        auto cp = llama_context_default_params();
        cp.n_ctx = o.context_tokens; cp.n_batch = 32;
        cp.n_ubatch = streaming ? std::max(1, std::min(16, o.prefill_slots / 8)) : 1;
        cp.n_threads = cp.n_threads_batch = o.cpu_threads; cp.n_seq_max = 1; cp.n_rs_seq = 0;
        cp.type_k = cp.type_v = GGML_TYPE_F16; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf = false; cp.abort_callback = abort; cp.abort_callback_data = user;
        if (o.trace_routing) {
            cp.cb_eval_user_data = result.get();
            cp.cb_eval = [](ggml_tensor * tensor, bool ask, void * data) {
                const bool selected = tensor->type == GGML_TYPE_I32 &&
                    std::string(ggml_get_name(tensor)).rfind("ffn_moe_topk-", 0) == 0 &&
                    ggml_nelements(tensor) <= 8192;
                if (ask) return selected;
                if (!selected) return true;
                auto * c = static_cast<flo_context *>(data);
                std::vector<int32_t> ids(ggml_nelements(tensor));
                ggml_backend_tensor_get(tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
                for (int32_t id : ids) { c->routing_digest ^= (uint32_t)id; c->routing_digest *= 1099511628211ULL; }
                c->routing_decisions += ids.size(); return true;
            };
        }
        result->context = llama_init_from_model(result->model, cp); require(result->context, "LES context allocation failed");
        result->vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(result->model));
        result->batch = (int)cp.n_ubatch; result->capacity = o.context_tokens;
        check_abort(abort, user); return result.release();
    } catch (const std::exception & e) { last_error = e.what(); return nullptr; }
}
void flo_close(flo_context * c) { delete c; }
int32_t flo_vocabulary_size(const flo_context * c) { return c ? c->vocabulary : 0; }
int flo_decode(flo_context * c, const int32_t * tokens, int32_t count, float * output, int32_t capacity) {
    try {
        require(c && !c->poisoned && tokens && output && count > 0 && count <= 32 && capacity == c->vocabulary &&
            count <= c->capacity - c->position, "Invalid LES decode request");
        for (int i = 0; i < count; ++i) require(tokens[i] >= 0 && tokens[i] < c->vocabulary, "Token outside vocabulary");
        for (int first = 0; first < count; first += c->batch) {
            check_abort(c->abort, c->user); const int n = std::min(c->batch, count - first);
            auto batch = llama_batch_get_one(const_cast<int32_t *>(tokens + first), n);
            require(llama_decode(c->context, batch) == 0, "LES forward failed"); c->position += n;
        }
        const float * logits = llama_get_logits_ith(c->context, -1); require(logits, "LES logits unavailable");
        for (int i = 0; i < capacity; ++i) require(std::isfinite(logits[i]), "Non-finite LES logit");
        std::copy(logits, logits + capacity, output); check_abort(c->abort, c->user); return 0;
    } catch (const std::exception & e) { if (c) c->poisoned = true; last_error = e.what(); return -1; }
}
void flo_reset(flo_context * c) {
    if (c && !c->poisoned) {
        llama_memory_clear(llama_get_memory(c->context), true); c->position = 0;
        c->routing_digest = 14695981039346656037ULL; c->routing_decisions = 0;
    }
}
flo_metrics flo_snapshot(const flo_context * c) {
    flo_metrics out{};
    if (c) { out.routing_digest = c->routing_digest; out.routing_decisions = c->routing_decisions; }
    if (c && c->context->flash_local_slots()) {
        const auto s = c->context->flash_local_slots()->counters_snapshot();
        out.io_bytes = s.all_bytes; out.io_calls = s.all_calls; out.misses = s.miss; out.selected_experts = s.unique;
        out.predicted = s.predicted; out.prediction_hits = s.prediction_hits; out.prefetch_bytes = s.prefetch_bytes;
        out.process_ns = s.process; out.install_ns = s.install;
    }
    flash_omni_metal_snapshot(&out.dispatches, &out.command_buffers, &out.gpu_ns); return out;
}
const char * flo_last_error() { return last_error.c_str(); }
const char * flo_revision() { return "96bbf61710d44421ea54e63f5194f9b17c20d877+flash-local-tensor-major-v1"; }
