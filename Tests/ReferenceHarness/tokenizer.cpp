#include "llama.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <vector>

static uint32_t read_u32(std::ifstream & input) {
    unsigned char bytes[4] = {};
    input.read(reinterpret_cast<char *>(bytes), 4);
    if (!input) throw std::runtime_error("Truncated corpus length");
    return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
           uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
}

// Vocabulary-only reference. This executable never creates a llama_context
// and its input GGUF contains zero tensors. No trained inference is possible.
int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    auto params = llama_model_default_params();
    params.vocab_only = true; params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE; params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    params.load_mtp = false;
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream input(argv[2], std::ios::binary);
    std::ofstream output(argv[3]);
    if (!input || !output) return 4;
    const auto count = read_u32(input);
    if (count > 1000) return 5;
    output << "{\"vocabularySize\":" << llama_vocab_n_tokens(vocab) << ",\"cases\":[";
    for (uint32_t i = 0; i < count; ++i) {
        const auto bytes = read_u32(input);
        if (bytes > 262144) return 6;
        std::string text(bytes, '\0'); input.read(text.data(), bytes);
        if (!input) return 7;
        std::vector<llama_token> tokens(bytes + 16);
        int n = llama_tokenize(vocab, text.data(), int(bytes), tokens.data(), int(tokens.size()), false, false);
        if (n < 0 || n > int(tokens.size())) return 8;
        if (i) output << ',';
        output << "{\"tokens\":[";
        std::string decoded;
        for (int j = 0; j < n; ++j) {
            if (j) output << ',';
            output << tokens[j];
            char piece[16384];
            const int length = llama_token_to_piece(vocab, tokens[j], piece, sizeof(piece), 0, true);
            if (length < 0) return 9;
            decoded.append(piece, length);
        }
        output << "],\"decodedHex\":\"";
        for (unsigned char byte : decoded) output << std::hex << std::setw(2) << std::setfill('0') << int(byte);
        output << std::dec << "\"}";
    }
    output << "],\"chats\":[";
    std::ifstream corpus_file(argv[4]);
    const auto corpus = common_json::parse(std::string(std::istreambuf_iterator<char>(corpus_file), {}));
    const char * source = llama_model_chat_template(model, nullptr);
    if (!source) return 11;
    jinja::lexer lexer;
    const auto program = jinja::parse_from_tokens(lexer.tokenize(source));
    bool first_chat = true;
    for (const auto & chat : corpus.at("chats")) {
        jinja::context context(source);
        jinja::global_from_json(context, chat, false);
        jinja::runtime runtime(context);
        auto parts = jinja::runtime::gather_string_parts(runtime.execute(program));
        const std::string rendered = parts->val_str.str();
        std::vector<llama_token> tokens(rendered.size() + 16);
        const int n = llama_tokenize(vocab, rendered.data(), int(rendered.size()), tokens.data(), int(tokens.size()), false, true);
        if (n < 0 || n > int(tokens.size())) return 12;
        if (!first_chat) output << ',';
        first_chat = false;
        output << "{\"rendered\":" << common_json::make(rendered).dump() << ",\"tokens\":[";
        for (int i = 0; i < n; ++i) { if (i) output << ','; output << tokens[i]; }
        output << "]}";
    }
    output << "]}\n";
    llama_model_free(model);
    return output.fail() ? 10 : 0;
}
