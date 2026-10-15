#include "llama.h"
#include "ggml-backend.h"
#include "json.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <filesystem>
#include <string>

struct Checkpoints {
    size_t position = 0;
    std::vector<common_json> steps = std::vector<common_json>(5, common_json::object());
};
static bool checkpoint(ggml_tensor * tensor, bool ask, void * user) {
    const std::string name = ggml_get_name(tensor);
    const bool selected = (name.rfind("l_last-", 0) == 0 || name == "result_norm") &&
        tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && ggml_nelements(tensor) <= 8192;
    if (ask) return selected;
    if (!selected) return true;
    auto & trace = *static_cast<Checkpoints *>(user);
    if (trace.position >= trace.steps.size()) return false;
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    trace.steps[trace.position][name] = values;
    return true;
}

// Synthetic tokens only. No tokenizer, user prompts, trained model or server.
int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 2;
    const bool trace_enabled = argc == 4 && std::string(argv[3]) == "--checkpoints";
    if (argc == 4 && (!trace_enabled || std::filesystem::file_size(argv[1]) > (32u << 20))) return 2;
    Checkpoints trace;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    mp.load_mtp = false;
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 32; cp.n_batch = 1; cp.n_ubatch = 1; cp.n_seq_max = 1;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (trace_enabled) { cp.cb_eval = checkpoint; cp.cb_eval_user_data = &trace; }
    llama_context * context = llama_init_from_model(model, cp);
    if (!context) { llama_model_free(model); return 4; }
    const int count = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens = {1, 2, 3, 15, 4};
    std::ofstream output(argv[2]);
    if (!output) return 5;
    output << std::setprecision(9) << "{\"tokens\":[1,2,3,15,4],\"logits\":[";
    for (size_t step = 0; step < tokens.size(); ++step) {
        trace.position = step;
        if (llama_decode(context, llama_batch_get_one(&tokens[step], 1)) != 0) return 6;
        const float * logits = llama_get_logits_ith(context, -1);
        if (!logits) return 7;
        if (step) output << ',';
        output << '[';
        for (int token = 0; token < count; ++token) {
            if (!std::isfinite(logits[token])) return 8;
            if (token) output << ',';
            output << logits[token];
        }
        output << ']';
    }
    output << ']';
    if (trace_enabled) output << ",\"checkpoints\":" << common_json(trace.steps).dump();
    output << "}\n";
    output.close();
    llama_free(context); llama_model_free(model); llama_backend_free();
    return output.fail() ? 9 : 0;
}
