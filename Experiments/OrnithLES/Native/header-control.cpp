// Host-only descriptor/layout control. No real weight payload, context or decode.
#include "llama.h"
#include "llama-model.h"
#include "gguf.h"
#include "LoaderDiagnostics.h"
#include <nlohmann/json.hpp>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
constexpr uint64_t source_bytes = 21713463040ULL, header_bytes = 10990848;
static flash_local_loader_diagnostics diagnostics;
static void require(bool condition, const char * reason) {
    if (!condition) throw std::runtime_error(reason);
}
struct metadata {
    ggml_context * tensors = nullptr;
    gguf_context * file = nullptr;
    ~metadata() { if (file) gguf_free(file); if (tensors) ggml_free(tensors); }
};
static uint32_t integer(gguf_context * file, const char * key) {
    const auto id = gguf_find_key(file, key);
    require(id >= 0 && gguf_get_kv_type(file, id) == GGUF_TYPE_UINT32, "Invalid pinned metadata integer");
    return gguf_get_val_u32(file, id);
}
static uint64_t sparse_extent(const fs::path & path, uint64_t size) {
    const int fd = open(path.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    require(fd >= 0, "Cannot open generated metadata-only file");
    const int resized = ftruncate(fd, size);
    struct stat status{}; const int measured = fstat(fd, &status); close(fd);
    require(resized == 0 && measured == 0 && uint64_t(status.st_size) == size, "Sparse extent creation failed");
    const uint64_t physical = uint64_t(status.st_blocks) * 512;
    require(physical <= 32ULL << 20, "Metadata-only file unexpectedly allocated weight storage");
    return physical;
}
static void configure(int cache, int depth) {
    const std::pair<const char *, int> options[] = {
        {"LLAMA_MOE_DECODE_WINDOW", cache}, {"LLAMA_MOE_PREFETCH_READ", depth},
        {"LLAMA_MOE_LOOKAHEAD_PREFETCH", depth > 0}, {"LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", depth},
        {"LLAMA_MOE_WINDOW_MAX_ROWS", 1}, {"LLAMA_MOE_SKIP_READ", 0}, {"LLAMA_MOE_DUMMY_READ", 0},
    };
    unsetenv("LLAMA_MOE_LAYER_WINDOWS"); unsetenv("LLAMA_MOE_PREWARM");
    for (const auto & item : options)
        require(setenv(item.first, std::to_string(item.second).c_str(), 1) == 0, "Cannot configure header-only control");
}
static void check_model(const fs::path & path, ggml_context * expected,
                        const std::string & manifest = "", int cache = 0, int depth = 0) {
    configure(cache, depth);
    auto params = llama_model_default_params();
    // Never create a context or dereference these intentionally absent weights.
    params.no_alloc = true; params.use_mmap = false; params.use_mlock = false;
    params.use_extra_bufts = false; params.n_gpu_layers = 0;
    if (!manifest.empty()) { params.moe_experts_path = manifest.c_str(); params.moe_slot_bank = 32; }
    diagnostics.begin();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path.c_str(), params), &llama_model_free);
    if (!model) throw std::runtime_error(diagnostics.describe("Metadata-only model load failed"));
    diagnostics.end();
    require(llama_model_n_layer(model.get()) == 41 && llama_model_n_embd(model.get()) == 2048,
            "Real model geometry changed");
    require(model->hparams.nextn_predict_layers == 1 && model->hparams.n_expert == 256 &&
            model->hparams.n_expert_used == 8, "Real routing or stored MTP geometry changed");
    for (int layer = 0; layer < 41; ++layer) {
        auto verify = [&](const char * family, const ggml_tensor * prefill, const ggml_tensor * decode) {
            const std::string name = "blk." + std::to_string(layer) + "." + family + ".weight";
            const auto * original = ggml_get_tensor(expected, name.c_str());
            require(original && prefill && prefill->type == original->type &&
                    prefill->ne[0] == original->ne[0] && prefill->ne[1] == original->ne[1] &&
                    prefill->nb[2] == ggml_nbytes(original) / 256 &&
                    prefill->ne[2] == (manifest.empty() ? 256 : 32), "Expert type, stride or prefill view changed");
            if (cache > 0) {
                require(decode && decode->type == original->type && decode->ne[2] == cache + depth &&
                        decode->nb[2] == prefill->nb[2], "Expert decode view changed");
            } else { require(!decode, "Unexpected decode allocation in metadata-only mode"); }
        };
        const auto & block = model->layers.at(layer);
        verify("ffn_gate_exps", block.ffn_gate_exps, block.ffn_gate_exps_dec);
        verify("ffn_up_exps", block.ffn_up_exps, block.ffn_up_exps_dec);
        verify("ffn_down_exps", block.ffn_down_exps, block.ffn_down_exps_dec);
    }
    const auto * vocabulary = llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(vocabulary) == 248320 && llama_vocab_eos(vocabulary) == 248046,
            "Real vocabulary geometry changed");
    llama_token tokens[8]{};
    const int n = llama_tokenize(vocabulary, "Hej", 3, tokens, 8, false, true);
    require(n == 2 && tokens[0] == 1465 && tokens[1] == 73, "Pinned tokenizer IDs changed");
}
int main(int argc, char ** argv) {
    try {
        require(argc == 4 && std::getenv("ORNITH_LES_CPU_CONTROL") &&
                std::string(std::getenv("ORNITH_LES_CPU_CONTROL")) == "1", "Explicit host header-control mode required");
        const fs::path header(argv[1]), root(argv[2]);
        require(fs::file_size(header) == header_bytes && fs::create_directory(root), "Pinned header or fresh output missing");
        metadata original, dense;
        original.file = gguf_init_from_file(header.c_str(), {true, &original.tensors});
        require(original.file && original.tensors && gguf_get_data_offset(original.file) == header_bytes,
                "Input must contain metadata only");
        require(integer(original.file, "qwen35moe.block_count") == 41 &&
                integer(original.file, "qwen35moe.expert_count") == 256 &&
                integer(original.file, "qwen35moe.expert_used_count") == 8 &&
                integer(original.file, "qwen35moe.nextn_predict_layers") == 1, "Pinned Ornith architecture changed");
        const fs::path source = root / "metadata-only-no-weights.gguf";
        fs::copy_file(header, source);
        const uint64_t source_physical = sparse_extent(source, source_bytes);
        const int descriptor = open(source.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        require(descriptor >= 0, "Cannot retain generated metadata source");
        dense.file = gguf_init_empty(); require(dense.file, "Dense metadata allocation failed");
        gguf_set_kv(dense.file, original.file);
        json entries = json::array();
        const std::regex routed("^blk\\.([0-9]+)\\.(ffn_(gate|up|down)_exps)\\.weight$");
        uint64_t original_end = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(original.file); ++i) {
            const std::string name = gguf_get_tensor_name(original.file, i);
            const auto * tensor = ggml_get_tensor(original.tensors, name.c_str());
            require(tensor, "Missing original tensor descriptor");
            const uint64_t bytes = ggml_nbytes(tensor);
            const uint64_t offset = header_bytes + gguf_get_tensor_offset(original.file, i);
            require(offset <= source_bytes && bytes <= source_bytes - offset, "Pinned tensor exceeds source extent");
            original_end = std::max(original_end, offset + bytes);
            std::smatch match;
            if (!std::regex_match(name, match, routed)) { gguf_add_tensor(dense.file, tensor); continue; }
            require(tensor->ne[2] == 256 && bytes % 256 == 0, "Pinned expert shape changed");
            entries.push_back({{"layer", std::stoi(match[1])}, {"tensor_family", match[2].str()},
                {"tensor_name", name}, {"quant_type", ggml_type_name(tensor->type)},
                {"exact_byte_length", bytes}, {"bytes_per_expert", bytes / 256}, {"expert_stride", bytes / 256},
                {"repacked_offset", offset}, {"repacked_file", "/dev/fd/" + std::to_string(descriptor)}});
        }
        require(original_end == source_bytes && entries.size() == 123, "Incomplete original shape coverage");
        const fs::path dense_path = root / "dense-metadata-only.gguf";
        require(gguf_write_to_file(dense.file, dense_path.c_str(), true), "Dense metadata write failed");
        uint64_t dense_end = gguf_get_meta_size(dense.file);
        for (int64_t i = 0; i < gguf_get_n_tensors(dense.file); ++i)
            dense_end = std::max(dense_end, gguf_get_meta_size(dense.file) + gguf_get_tensor_offset(dense.file, i) + gguf_get_tensor_size(dense.file, i));
        const uint64_t dense_physical = sparse_extent(dense_path, dense_end);
        const fs::path manifest_path = root / "manifest.json";
        json manifest{{"schema_version", 1}, {"experts_kind", "omni_moe_gguf"}, {"layout", "gguf_tensor_major"},
            {"model", {{"expert_count", 256}, {"expert_used_count", 8}}}, {"entries", entries}};
        { std::ofstream out(manifest_path); out << manifest.dump(); require(bool(out), "Manifest write failed"); }
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (text) fputs(text, stderr); diagnostics.record(level, text);
        }, nullptr);
        llama_backend_init();
        check_model(source, original.tensors);
        check_model(dense_path, original.tensors, manifest_path.string());
        check_model(dense_path, original.tensors, manifest_path.string(), 32, 0);
        check_model(dense_path, original.tensors, manifest_path.string(), 28, 4);
        close(descriptor);
        struct rusage usage{}; require(getrusage(RUSAGE_SELF, &usage) == 0, "Host footprint unavailable");
        require(usage.ru_maxrss < 1LL << 30, "Metadata-only control exceeded 1 GiB RSS");
        json report{{"metadataOnly", true}, {"trainedWeightPayloadPresent", false}, {"inferenceExecuted", false},
            {"fullModelNumericsVerified", false}, {"metalExecutionVerified", false}, {"physicalPhone", false},
            {"storedLayers", 41}, {"mainLayers", 40}, {"experts", 256}, {"topK", 8},
            {"routedTensorShapesChecked", entries.size()}, {"modelLoaderVariantsChecked", 4},
            {"allTensorShapesChecked", gguf_get_n_tensors(original.file)}, {"tokenizerIDsIdentical", true},
            {"sourceLogicalBytes", source_bytes}, {"denseLogicalBytes", dense_end},
            {"allocatedStorageBytes", source_physical + dense_physical}, {"hostPeakRSSBytes", usage.ru_maxrss}};
        std::ofstream out(argv[3]); out << report.dump(2) << '\n'; require(bool(out), "Report write failed");
        std::cout << "ORNITH_REAL_HEADER fullGeometry=true metadataOnly=true noWeightPayload=true inferenceExecuted=false\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
