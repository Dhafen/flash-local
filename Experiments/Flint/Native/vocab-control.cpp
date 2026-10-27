#include "llama.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("GGUF-header-only requests.json");
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.vocab_only = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("Vocab-only load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        nlohmann::json input; std::ifstream stream(argv[2]); stream >> input;
        auto output = nlohmann::json::array();
        for (const auto & value : input) {
            const auto text = value.get<std::string>();
            const int needed = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
            std::vector<llama_token> ids(size_t(-needed));
            const int n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, true);
            if (n < 0 || n != int(ids.size())) throw std::runtime_error("Tokenization length mismatch");
            output.push_back(ids);
        }
        std::cout << output.dump() << std::endl; return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << std::endl; return 1; }
}
