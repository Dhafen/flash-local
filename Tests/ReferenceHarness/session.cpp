#include "FlashLlamaTokenizer.h"
#include "llama.h"
#include "json.h"
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

// Complete text->tokens->synthetic graph->greedy tokens reference. The caller
// admits only our <1 MB deterministic random-weight fixture, not checkpoints.
int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    std::ifstream manifest_file(argv[2]);
    const auto manifest = common_json::parse(std::string(std::istreambuf_iterator<char>(manifest_file), {}));
    const auto messages = manifest.at("messages").dump();
    const int maximum = manifest.at("maximumNewTokens").get<int>();
    if (maximum < 1 || maximum > 16) return 3;
    const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 4;
    auto tokenizer = flt_load_vocabulary(fd); close(fd);
    if (!tokenizer) return 5;
    int bytes = -flt_render_chat(tokenizer, messages.data(), int(messages.size()), nullptr, 0);
    if (bytes < 1 || bytes > 8192) return 6;
    std::vector<char> rendered(bytes);
    if (flt_render_chat(tokenizer, messages.data(), int(messages.size()), rendered.data(), bytes) != bytes) return 7;
    std::vector<int32_t> prompt(bytes + 16);
    const int count = flt_encode(tokenizer, rendered.data(), bytes, true, prompt.data(), int(prompt.size()));
    if (count < 1 || count + maximum > 128) return 8;
    prompt.resize(count);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0; mp.load_mode = LLAMA_LOAD_MODE_NONE; mp.lazy_mode = LLAMA_LAZY_MODE_OFF; mp.load_mtp = false;
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 9;
    auto cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 1; cp.n_ubatch = 1; cp.n_seq_max = 1;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto context = llama_init_from_model(model, cp);
    if (!context) return 10;
    for (auto token : prompt) if (llama_decode(context, llama_batch_get_one(&token, 1))) return 11;
    auto sampler = llama_sampler_init_greedy();
    std::vector<int32_t> generated;
    std::string decoded;
    bool eog = false;
    double minimum_margin = std::numeric_limits<double>::max();
    for (int step = 0; step < maximum; ++step) {
        const float * logits = llama_get_logits_ith(context, -1);
        if (!logits) return 12;
        float first = -std::numeric_limits<float>::infinity(), second = first;
        for (int i = 0; i < flt_vocabulary_size(tokenizer); ++i) {
            if (!std::isfinite(logits[i])) return 13;
            if (logits[i] > first) { second = first; first = logits[i]; }
            else if (logits[i] > second) second = logits[i];
        }
        minimum_margin = std::min(minimum_margin, double(first) - double(second));
        auto token = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(llama_model_get_vocab(model), token)) { eog = true; break; }
        generated.push_back(token); llama_sampler_accept(sampler, token);
        char piece[16384];
        const int n = llama_token_to_piece(llama_model_get_vocab(model), token, piece, sizeof(piece), 0, true);
        if (n < 0) return 14;
        decoded.append(piece, n);
        if (step + 1 < maximum && llama_decode(context, llama_batch_get_one(&token, 1))) return 15;
    }
    auto result = common_json::object();
    result["promptTokenIDs"] = common_json::array(); result["generatedTokenIDs"] = common_json::array();
    for (auto token : prompt) result["promptTokenIDs"].push_back(int(token));
    for (auto token : generated) result["generatedTokenIDs"].push_back(int(token));
    result["stoppedAtEOG"] = eog; result["minimumWinnerMargin"] = minimum_margin;
    std::string hex; const char digits[] = "0123456789abcdef";
    for (unsigned char c : decoded) { hex += digits[c >> 4]; hex += digits[c & 15]; }
    result["decodedHex"] = hex;
    std::ofstream output(argv[3]); output << result.dump(2) << '\n';
    llama_sampler_free(sampler); llama_free(context); llama_model_free(model);
    llama_backend_free(); flt_free_vocabulary(tokenizer);
    return output.fail() ? 16 : 0;
}
