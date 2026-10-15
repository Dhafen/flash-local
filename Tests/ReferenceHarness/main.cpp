#include "llama.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// Synthetic tokens only. No tokenizer, user prompts, trained model or server.
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 32; cp.n_batch = 1; cp.n_ubatch = 1; cp.n_seq_max = 1;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * context = llama_init_from_model(model, cp);
    if (!context) { llama_model_free(model); return 4; }
    const int count = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens = {1, 2, 3, 15, 4};
    std::ofstream output(argv[2]);
    if (!output) return 5;
    output << std::setprecision(9) << "{\"tokens\":[1,2,3,15,4],\"logits\":[";
    for (size_t step = 0; step < tokens.size(); ++step) {
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
    output << "]}\n";
    output.close();
    llama_free(context); llama_model_free(model); llama_backend_free();
    return output.fail() ? 9 : 0;
}
