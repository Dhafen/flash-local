// Synthetic graph/streaming control. This is not a trained-model benchmark.
#include "llama.h"
#include "llama-context.h"
#include "llama-moe-slot.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <memory>
#include <cmath>
#include <filesystem>

extern "C" void ggml_flint_set_q4_reference(bool enabled);

int main(int argc, char ** argv) {
    try {
        if (argc != 5 && argc != 6) throw std::runtime_error("model.gguf manifest-or-- steps physical-batch [q4-f32-reference]");
        const bool q4_reference = argc == 6 && std::string(argv[5]) == "q4-f32-reference";
        if (argc == 6 && !q4_reference) throw std::runtime_error("Unknown diagnostic precision policy");
        const int steps = std::stoi(argv[3]), physical = std::stoi(argv[4]);
        if (steps < 1 || steps > 512 || physical < 1 || physical > 32 || std::filesystem::file_size(argv[1]) > (64u << 20))
            throw std::runtime_error("Invalid control geometry; only small random fixtures under 64 MiB are admitted");
        setenv("ORNITH_LES_CPU_CONTROL", "1", 1);
        for (const char * key : {"LLAMA_MOE_DECODE_WINDOW", "LLAMA_MOE_LOOKAHEAD_PREFETCH", "LLAMA_MOE_PREFETCH_READ",
             "LLAMA_MOE_SKIP_READ", "LLAMA_MOE_DUMMY_READ", "LLAMA_MOE_MATMUL_SPLIT", "LLAMA_MOE_F_NOCACHE"}) setenv(key, "0", 1);
        unsetenv("LLAMA_MOE_LAYER_WINDOWS");
        ggml_flint_set_q4_reference(q4_reference);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        // CPU synthetic alloc-only buffers need ordinary owned allocation.
        // The Metal path retains mmap for fixed weights.
        mp.use_mmap = false;
        mp.use_mlock = false;
        const bool streaming = std::string(argv[2]) != "-";
        if (streaming) { mp.moe_experts_path = argv[2]; mp.moe_slot_bank = std::min(256, physical * 6); }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("Model load failed");
        llama_context_params cp = llama_context_default_params();
        struct trace_state { int position = 0; nlohmann::json rows = nlohmann::json::array(); } trace;
        cp.cb_eval_user_data = &trace;
        cp.cb_eval = [](ggml_tensor * tensor, bool ask, void * data) {
            const std::string name = ggml_get_name(tensor);
            const std::string prefix = "ffn_moe_topk-";
            const bool topk = name.rfind(prefix, 0) == 0;
            if (ask) return topk;
            if (!topk) return true;
            if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 6 || tensor->ne[2] != 1 || tensor->ne[3] != 1) return false;
            auto & state = *static_cast<trace_state *>(data);
            for (int row = 0; row < tensor->ne[1]; ++row) {
                std::vector<int32_t> ids(6);
                ggml_backend_tensor_get(tensor, ids.data(), size_t(row) * tensor->nb[1], 6 * sizeof(int32_t));
                state.rows.push_back({{"position", state.position + row}, {"layer", std::stoi(name.substr(prefix.size()))}, {"experts", ids}});
            }
            return true;
        };
        cp.n_ctx = 512; cp.n_batch = 32; cp.n_ubatch = physical; cp.n_seq_max = 1;
        cp.n_threads = 2; cp.n_threads_batch = 2; cp.swa_full = false;
        cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("Context init failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        nlohmann::json rows = nlohmann::json::array();
        for (int first = 0; first < steps;) {
            const int count = std::min(physical, steps - first);
            trace.position = first;
            llama_batch batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = 1 + (first + i) % (vocab - 1);
                batch.pos[i] = first + i; batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0; batch.logits[i] = true;
            }
            const int status = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            if (status) throw std::runtime_error("llama_decode failed: " + std::to_string(status));
            for (int i = 0; i < count; ++i) {
                const float * logits = llama_get_logits_ith(ctx.get(), i);
                std::vector<float> row(logits, logits + vocab);
                for (float x : row) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite logits");
                rows.push_back(row);
            }
            first += count;
        }
        uint64_t bytes = 0, reads = 0;
        if (auto * slots = ctx->flash_local_slots()) { auto s = slots->counters_snapshot(); bytes = s.all_bytes; reads = s.all_calls; }
        std::cout << nlohmann::json{{"synthetic", true}, {"streaming", streaming}, {"q4F32Reference", q4_reference}, {"physicalBatch", physical},
            {"steps", steps}, {"ioBytes", bytes}, {"ioCalls", reads}, {"routes", trace.rows}, {"logits", rows}}.dump() << std::endl;
        return 0;
    } catch (const std::exception & e) { std::cerr << "Flint control: " << e.what() << std::endl; return 1; }
}
