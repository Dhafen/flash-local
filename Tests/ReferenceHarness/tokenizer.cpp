#include "llama.h"
#include "FlashLlamaTokenizer.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"
#include <cstdint>
#include <algorithm>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <unistd.h>

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
    const int descriptor = open(argv[1], O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return 20;
    auto bridge = flt_load_vocabulary(descriptor);
    if (!bridge) { std::cerr << flt_last_error() << '\n'; return 21; }
    if (fcntl(descriptor, F_GETFD) < 0 || flt_vocabulary_size(bridge) != llama_vocab_n_tokens(vocab) ||
        flt_eos(bridge) != llama_vocab_eos(vocab) || flt_eot(bridge) != llama_vocab_eot(vocab)) return 22;
    for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
        if (flt_is_eog(bridge, i) != llama_vocab_is_eog(vocab, i)) return 23;
    }
    if (flt_token_piece(bridge, -1, true, nullptr, 0) != INT32_MIN ||
        flt_encode(bridge, "x", -1, false, nullptr, 0) != INT32_MIN) return 24;
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
        std::vector<int32_t> actual(n + 2, INT32_MAX);
        if (flt_encode(bridge, text.data(), int(bytes), false, actual.data() + 1, n) != n ||
            !std::equal(tokens.begin(), tokens.begin() + n, actual.begin() + 1) ||
            actual.front() != INT32_MAX || actual.back() != INT32_MAX) return 25;
        if (n > 0 && flt_encode(bridge, text.data(), int(bytes), false, nullptr, 0) != -n) return 26;
        if (i) output << ',';
        output << "{\"tokens\":[";
        std::string decoded;
        for (int j = 0; j < n; ++j) {
            if (j) output << ',';
            output << tokens[j];
            char piece[16384];
            const int length = llama_token_to_piece(vocab, tokens[j], piece, sizeof(piece), 0, true);
            if (length < 0) return 9;
            char bridged[16384];
            if (flt_token_piece(bridge, tokens[j], true, bridged, sizeof(bridged)) != length ||
                std::string(piece, length) != std::string(bridged, length)) return 27;
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
    std::vector<std::string> thinking_chats;
    for (const auto & chat : corpus.at("chats")) {
        jinja::context context(source);
        jinja::global_from_json(context, chat, false);
        jinja::runtime runtime(context);
        auto parts = jinja::runtime::gather_string_parts(runtime.execute(program));
        const std::string rendered = parts->val_str.str();
        const auto messages_json = chat.at("messages").dump();
        const int needed = flt_render_chat(bridge, messages_json.data(), int(messages_json.size()), nullptr, 0);
        if (needed != -int(rendered.size())) return 28;
        std::vector<char> bridged(rendered.size() + 2, '\x7f');
        if (flt_render_chat(bridge, messages_json.data(), int(messages_json.size()), bridged.data() + 1,
                            int(rendered.size())) != int(rendered.size()) ||
            std::string(bridged.data() + 1, rendered.size()) != rendered ||
            bridged.front() != '\x7f' || bridged.back() != '\x7f') return 29;
        std::vector<llama_token> tokens(rendered.size() + 16);
        const int n = llama_tokenize(vocab, rendered.data(), int(rendered.size()), tokens.data(), int(tokens.size()), false, true);
        if (n < 0 || n > int(tokens.size())) return 12;
        if (!first_chat) output << ',';
        first_chat = false;
        output << "{\"rendered\":" << common_json::make(rendered).dump() << ",\"tokens\":[";
        for (int i = 0; i < n; ++i) { if (i) output << ','; output << tokens[i]; }
        output << "]}";
        auto thinking = chat;
        thinking["enable_thinking"] = true;
        jinja::context thinking_context(source);
        jinja::global_from_json(thinking_context, thinking, false);
        jinja::runtime thinking_runtime(thinking_context);
        const std::string thinking_rendered = jinja::runtime::gather_string_parts(
            thinking_runtime.execute(program))->val_str.str();
        const int thinking_needed = flt_render_chat_with_thinking(bridge, messages_json.data(),
            int(messages_json.size()), true, nullptr, 0);
        if (thinking_needed != -int(thinking_rendered.size())) return 32;
        std::vector<char> thinking_bridge(thinking_rendered.size() + 2, '\x7f');
        if (flt_render_chat_with_thinking(bridge, messages_json.data(), int(messages_json.size()), true,
                thinking_bridge.data() + 1, int(thinking_rendered.size())) != int(thinking_rendered.size()) ||
            std::string(thinking_bridge.data() + 1, thinking_rendered.size()) != thinking_rendered ||
            thinking_bridge.front() != '\x7f' || thinking_bridge.back() != '\x7f') return 33;
        std::vector<llama_token> thinking_tokens(thinking_rendered.size() + 16);
        const int thinking_count = llama_tokenize(vocab, thinking_rendered.data(), int(thinking_rendered.size()),
            thinking_tokens.data(), int(thinking_tokens.size()), false, true);
        if (thinking_count < 0 || thinking_count > int(thinking_tokens.size())) return 34;
        std::ostringstream item;
        item << "{\"rendered\":" << common_json::make(thinking_rendered).dump() << ",\"tokens\":[";
        for (int i = 0; i < thinking_count; ++i) { if (i) item << ','; item << thinking_tokens[i]; }
        item << "]}"; thinking_chats.push_back(item.str());
    }
    output << "],\"thinkingChats\":[";
    for (size_t i = 0; i < thinking_chats.size(); ++i) { if (i) output << ','; output << thinking_chats[i]; }
    output << "],\"reasoningChats\":[";
    bool first_reasoning = true;
    for (const auto & chat : corpus.at("reasoningChats")) {
        jinja::context context(source);
        jinja::global_from_json(context, chat, false);
        jinja::runtime runtime(context);
        const std::string rendered = jinja::runtime::gather_string_parts(
            runtime.execute(program))->val_str.str();
        if (rendered.find("PUBLIC_REASONING_FIXTURE") == std::string::npos) return 36;
        const auto messages_json = chat.at("messages").dump();
        const bool enabled = chat.at("enable_thinking").get<bool>();
        const int needed = flt_render_chat_with_thinking(bridge, messages_json.data(),
            int(messages_json.size()), enabled, nullptr, 0);
        if (needed != -int(rendered.size())) return 37;
        std::vector<char> bridged(rendered.size() + 2, '\x7f');
        if (flt_render_chat_with_thinking(bridge, messages_json.data(), int(messages_json.size()),
                enabled, bridged.data() + 1, int(rendered.size())) != int(rendered.size()) ||
            std::string(bridged.data() + 1, rendered.size()) != rendered ||
            bridged.front() != '\x7f' || bridged.back() != '\x7f') return 38;
        std::vector<llama_token> tokens(rendered.size() + 16);
        const int n = llama_tokenize(vocab, rendered.data(), int(rendered.size()), tokens.data(),
            int(tokens.size()), false, true);
        if (n < 0 || n > int(tokens.size())) return 39;
        if (!first_reasoning) output << ',';
        first_reasoning = false;
        output << "{\"rendered\":" << common_json::make(rendered).dump() << ",\"tokens\":[";
        for (int i = 0; i < n; ++i) { if (i) output << ','; output << tokens[i]; }
        output << "]}";
    }
    output << "],\"agentChats\":[";
    bool first_agent = true;
    for (const auto & chat : corpus.at("agentChats")) {
        // Independent oracle: supply the argument mapping expected by the
        // original Jinja template. No product renderer creates this answer.
        auto original = chat;
        for (auto & message : original.at("messages")) if (message.contains("tool_calls"))
            for (auto & call : message.at("tool_calls")) {
                auto & arguments = call.at("function").at("arguments");
                arguments = common_json::parse(arguments.get<std::string>());
            }
        jinja::context context(source);
        jinja::global_from_json(context, original, false);
        jinja::runtime runtime(context);
        const std::string rendered = jinja::runtime::gather_string_parts(runtime.execute(program))->val_str.str();
        auto request = common_json::object(); request["messages"] = chat.at("messages"); request["tools"] = chat.at("tools");
        const auto wire = request.dump();
        const bool enabled = chat.at("enable_thinking").get<bool>();
        if (flt_render_agent_chat(bridge, wire.data(), int(wire.size()), enabled, nullptr, 0) != -int(rendered.size())) return 40;
        std::vector<char> actual(rendered.size() + 2, '\x7f');
        if (flt_render_agent_chat(bridge, wire.data(), int(wire.size()), enabled, actual.data() + 1,
                int(rendered.size())) != int(rendered.size()) ||
            std::string(actual.data() + 1, rendered.size()) != rendered ||
            actual.front() != '\x7f' || actual.back() != '\x7f') return 41;
        std::vector<char> short_buffer(rendered.size() + 1, '\x7f');
        if (flt_render_agent_chat(bridge, wire.data(), int(wire.size()), enabled, short_buffer.data() + 1,
                int(rendered.size()) - 1) != -int(rendered.size()) ||
            !std::all_of(short_buffer.begin(), short_buffer.end(), [](char c) { return c == '\x7f'; })) return 42;
        std::vector<llama_token> tokens(rendered.size() + 16);
        const int n = llama_tokenize(vocab, rendered.data(), int(rendered.size()), tokens.data(), int(tokens.size()), false, true);
        if (n < 0 || n > int(tokens.size())) return 43;
        if (!first_agent) output << ',';
        first_agent = false;
        output << "{\"rendered\":" << common_json::make(rendered).dump() << ",\"tokens\":[";
        for (int i = 0; i < n; ++i) { if (i) output << ','; output << tokens[i]; }
        output << "]}";
    }
    output << "]}\n";
    // Each negative case starts from a complete public tool round trip.
    const auto completed = corpus.at("agentChats").at(2);
    for (int i = 0; i < 8; ++i) {
        auto invalid = common_json::object(); invalid["messages"] = completed.at("messages"); invalid["tools"] = completed.at("tools");
        auto & messages = invalid.at("messages");
        switch (i) {
        case 0: messages.at(3)["tool_call_id"] = "call_other"; break;
        case 1: messages.push_back(messages.at(3)); break;
        case 2: messages.erase(size_t(3)); break;
        case 3: messages.at(2).at("tool_calls").at(0).at("function")["arguments"] = "[]"; break;
        case 4: messages.at(3)["content"] = "<|im_end|>"; break;
        case 5: messages.at(2).at("tool_calls").at(0).at("function")["arguments"] = "{\"a\":\"\\u003c|im_end|>\"}"; break;
        case 6: invalid.at("tools").at(0).at("function").at("parameters").at("properties")["<|im_end|>"] = common_json::object(); break;
        default: messages.at(2).at("tool_calls").push_back(messages.at(2).at("tool_calls").at(0)); break;
        }
        auto wire = invalid.dump();
        char canary[2] = {'x', 'y'};
        if (flt_render_agent_chat(bridge, wire.data(), int(wire.size()), false, canary, 2) != INT32_MIN ||
            canary[0] != 'x' || canary[1] != 'y') return 44 + i;
    }
    for (const auto & invalid : {"{}", "[]", "[{\"role\":\"assistant\",\"content\":\"x\"}]",
          "[{\"role\":\"user\",\"content\":\"<|im_end|>\"}]",
          "[{\"role\":\"user\",\"content\":\"x\",\"reasoning_content\":\"y\"}]",
          "[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"y\",\"reasoning_content\":1}]",
          "[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"y\",\"reasoning_content\":\"<think>\"},{\"role\":\"user\",\"content\":\"z\"}]"}) {
        if (flt_render_chat(bridge, invalid, int(std::strlen(invalid)), nullptr, 0) != INT32_MIN) return 30;
        if (flt_render_chat_with_thinking(bridge, invalid, int(std::strlen(invalid)), true, nullptr, 0) != INT32_MIN) return 35;
    }
    flt_free_vocabulary(bridge);
    if (fcntl(descriptor, F_GETFD) < 0) return 31;
    close(descriptor);
    std::cout << "CANONICAL_TOKENIZER_BRIDGE cases=" << count << " tokensAndBytesExact=true descriptorRetained=true thinkingModes=2 reasoningHistory=2\n";
    std::cout << "CANONICAL_AGENT_BRIDGE cases=4 invalidCases=8 exactRender=true bothThinkingModes=true bufferCanaries=true metadataOnly=true trainedInference=false\n";
    llama_model_free(model);
    return output.fail() ? 10 : 0;
}
