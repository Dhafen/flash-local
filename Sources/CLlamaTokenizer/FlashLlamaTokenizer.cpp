#include "FlashLlamaTokenizer.h"
#include "llama.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include <vector>

struct flt_vocabulary {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string chat_template;
    std::unique_ptr<jinja::program> program;
    std::vector<std::string> reserved;
    ~flt_vocabulary() { if (model) llama_model_free(model); }
};
static thread_local std::string last_error;
static int32_t fail(const char * message) { last_error = message; return INT32_MIN; }
const char * flt_last_error() { return last_error.c_str(); }

flt_vocabulary * flt_load_vocabulary(int descriptor) {
    try {
        const int duplicate = fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
        if (duplicate < 0) { fail("Cannot duplicate verified vocabulary descriptor"); return nullptr; }
        FILE * file = fdopen(duplicate, "rb");
        if (!file) { close(duplicate); fail("Cannot open vocabulary FILE"); return nullptr; }
        std::unique_ptr<FILE, decltype(&fclose)> owned_file(file, &fclose);
        if (fseek(file, 0, SEEK_SET) != 0) { fail("Cannot seek verified GGUF"); return nullptr; }
        auto params = llama_model_default_params();
        params.vocab_only = true; params.n_gpu_layers = 0; params.load_mtp = false;
        params.load_mode = LLAMA_LOAD_MODE_NONE; params.lazy_mode = LLAMA_LAZY_MODE_OFF;
        auto result = std::make_unique<flt_vocabulary>();
        result->model = llama_model_load_from_file_ptr(file, params);
        if (!result->model) { fail("llama.cpp could not load GGUF vocabulary"); return nullptr; }
        result->vocab = llama_model_get_vocab(result->model);
        char pre[64] = {}, kind[64] = {};
        llama_model_meta_val_str(result->model, "tokenizer.ggml.pre", pre, sizeof(pre));
        llama_model_meta_val_str(result->model, "tokenizer.ggml.model", kind, sizeof(kind));
        const int count = llama_vocab_n_tokens(result->vocab);
        if (std::strcmp(pre, "qwen35") || std::strcmp(kind, "gpt2") || count < 256 || count > 300000) {
            fail("Expected bounded Qwen35 byte-BPE vocabulary"); return nullptr;
        }
        if (const char * source = llama_model_chat_template(result->model, nullptr)) {
            result->chat_template = source;
            if (result->chat_template.size() > 32768) { fail("Chat template exceeds 32 KiB"); return nullptr; }
            jinja::lexer lexer;
            result->program = std::make_unique<jinja::program>(jinja::parse_from_tokens(lexer.tokenize(source)));
        }
        for (int i = 0; i < count; ++i) {
            const auto attr = llama_vocab_get_attr(result->vocab, i);
            if (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                const char * text = llama_vocab_get_text(result->vocab, i);
                if (text && *text) result->reserved.emplace_back(text);
            }
        }
        last_error.clear(); return result.release();
    } catch (const std::exception & error) { fail(error.what()); return nullptr; }
      catch (...) { fail("Unknown vocabulary load failure"); return nullptr; }
}
void flt_free_vocabulary(flt_vocabulary * vocabulary) { delete vocabulary; }
int32_t flt_vocabulary_size(const flt_vocabulary * v) { return v ? llama_vocab_n_tokens(v->vocab) : 0; }
int32_t flt_eos(const flt_vocabulary * v) { return v ? llama_vocab_eos(v->vocab) : -1; }
int32_t flt_eot(const flt_vocabulary * v) { return v ? llama_vocab_eot(v->vocab) : -1; }
bool flt_is_eog(const flt_vocabulary * v, int32_t token) { return v && llama_vocab_is_eog(v->vocab, token); }
const char * flt_chat_template(const flt_vocabulary * v) { return v ? v->chat_template.c_str() : nullptr; }

int32_t flt_encode(const flt_vocabulary * v, const char * text, int32_t bytes,
                   bool parse_special, int32_t * output, int32_t capacity) {
    if (!v || bytes < 0 || bytes > 524288 || (!text && bytes) || capacity < 0 ||
        capacity > 131072 || (!output && capacity)) return fail("Invalid bounded tokenizer request");
    try {
        return llama_tokenize(v->vocab, text ? text : "", bytes, output, capacity, false, parse_special);
    } catch (const std::exception & error) { return fail(error.what()); }
      catch (...) { return fail("Unknown tokenization failure"); }
}
int32_t flt_token_piece(const flt_vocabulary * v, int32_t token, bool special,
                        char * output, int32_t capacity) {
    if (!v || token < 0 || token >= flt_vocabulary_size(v) || capacity < 0 || capacity > 16384 ||
        (!output && capacity)) return fail("Invalid token piece request");
    try { return llama_token_to_piece(v->vocab, token, output, capacity, 0, special); }
    catch (const std::exception & error) { return fail(error.what()); }
    catch (...) { return fail("Unknown detokenization failure"); }
}
int32_t flt_render_chat(const flt_vocabulary * v, const char * messages_json,
                        int32_t bytes, char * output, int32_t capacity) {
    return flt_render_chat_with_thinking(v, messages_json, bytes, false, output, capacity);
}
int32_t flt_render_chat_with_thinking(const flt_vocabulary * v, const char * messages_json,
                        int32_t bytes, bool enable_thinking, char * output, int32_t capacity) {
    if (!v || !v->program || !messages_json || bytes < 0 || bytes > 1048576 || capacity < 0 ||
        capacity > 524288 || (!output && capacity)) return fail("Invalid bounded chat request");
    try {
        auto messages = common_json::parse(std::string(messages_json, bytes));
        if (!messages.is_array() || messages.empty() || messages.size() > 4096) return fail("Chat message count");
        std::string expected = "user"; size_t total = 0, index = 0; bool reasoning_history = false;
        for (const auto & message : messages) {
            const bool has_reasoning = message.is_object() && message.contains("reasoning_content");
            if (!message.is_object() || message.size() != size_t(has_reasoning ? 3 : 2) || !message.contains("role") ||
                !message.contains("content") || !message.at("role").is_string() ||
                !message.at("content").is_string()) return fail("Chat needs typed text messages");
            auto role = message.at("role").get<std::string>();
            auto content = message.at("content").get<std::string>();
            total += content.size();
            std::string reasoning;
            if (has_reasoning) {
                if (role != "assistant" || !message.at("reasoning_content").is_string()) return fail("Invalid assistant reasoning");
                reasoning = message.at("reasoning_content").get<std::string>();
                total += reasoning.size(); reasoning_history = reasoning_history || !reasoning.empty();
            }
            if (total > 262144) return fail("Chat content exceeds 256 KiB");
            for (const auto & marker : v->reserved) {
                if (content.find(marker) != std::string::npos || reasoning.find(marker) != std::string::npos)
                    return fail("Message contains structural tokenizer markup");
            }
            if (index++ == 0 && role == "system") continue;
            if (role != expected) return fail("Chat roles must alternate after the optional system message");
            expected = role == "user" ? "assistant" : "user";
        }
        if (expected != "assistant") return fail("Chat needs a final user turn");
        auto input = common_json::object(); input["messages"] = messages;
        input["enable_thinking"] = enable_thinking; input["preserve_thinking"] = reasoning_history;
        input["add_generation_prompt"] = true;
        jinja::context context(v->chat_template);
        jinja::global_from_json(context, input, false);
        jinja::runtime runtime(context);
        const auto result = jinja::runtime::gather_string_parts(runtime.execute(*v->program))->val_str.str();
        if (result.size() > 524288) return fail("Rendered chat exceeds 512 KiB");
        if (result.size() > size_t(capacity)) return -int32_t(result.size());
        if (!result.empty()) std::memcpy(output, result.data(), result.size());
        return int32_t(result.size());
    } catch (const std::exception & error) { return fail(error.what()); }
      catch (...) { return fail("Unknown chat rendering failure"); }
}

int32_t flt_render_agent_chat(const flt_vocabulary * v, const char * request_json,
                        int32_t bytes, bool enable_thinking, char * output, int32_t capacity) {
    if (!v || !v->program || !request_json || bytes < 0 || bytes > 1048576 || capacity < 0 ||
        capacity > 524288 || (!output && capacity)) return fail("Invalid bounded agent request");
    try {
        auto input = common_json::parse(std::string(request_json, bytes));
        if (!input.is_object() || input.size() != 2 || !input.contains("messages") || !input.contains("tools"))
            return fail("Agent request needs messages and tools");
        auto & messages = input.at("messages"); auto & tools = input.at("tools");
        if (!messages.is_array() || messages.empty() || messages.size() > 4096 || !tools.is_array() || tools.size() > 32)
            return fail("Agent message/tool count");
        size_t total = 0, nodes = 0;
        auto inspect = [&](auto && self, const common_json & value, int depth) -> void {
            if (depth > 32 || ++nodes > 32768) throw std::runtime_error("Agent JSON nesting/node bound");
            if (value.is_string()) {
                auto text = value.get<std::string>(); total += text.size();
                if (total > 262144) throw std::runtime_error("Agent content exceeds 256 KiB");
                for (const auto & marker : v->reserved) if (text.find(marker) != std::string::npos)
                    throw std::runtime_error("Agent text contains structural tokenizer markup");
            } else if (value.is_object()) {
                for (auto item = value.begin(); item != value.end(); ++item) {
                    self(self, common_json::make(item.key()), depth + 1);
                    self(self, item.value(), depth + 1);
                }
            } else if (value.is_array()) {
                for (const auto & item : value) self(self, item, depth + 1);
            }
        };
        inspect(inspect, input, 0);
        auto named = [](const std::string & text, size_t maximum) {
            return !text.empty() && text.size() <= maximum && std::all_of(text.begin(), text.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
            });
        };
        std::unordered_set<std::string> names, seen, pending;
        for (const auto & tool : tools) {
            if (!tool.is_object() || tool.size() != 2 || tool.value("type", "") != "function" || !tool.contains("function"))
                return fail("Expected typed function definitions");
            const auto & function = tool.at("function");
            if (!function.is_object() || !function.contains("name") || !function.at("name").is_string() ||
                !function.contains("parameters") || !function.at("parameters").is_object()) return fail("Invalid function definition");
            auto name = function.at("name").get<std::string>();
            if (!named(name, 64) || !names.insert(name).second) return fail("Invalid or duplicate function name");
        }
        std::string expected = "user", last; size_t index = 0; bool preserve = false;
        for (const auto & message : messages) {
            if (!message.is_object() || !message.contains("role") || !message.at("role").is_string() ||
                !message.contains("content") || !message.at("content").is_string()) return fail("Agent needs typed message content");
            for (auto field = message.begin(); field != message.end(); ++field)
                if (field.key() != "role" && field.key() != "content" && field.key() != "reasoning_content" &&
                    field.key() != "tool_calls" && field.key() != "tool_call_id") return fail("Unexpected agent message field");
            auto role = message.at("role").get<std::string>(); last = role;
            if (message.contains("reasoning_content")) {
                if (role != "assistant" || !message.at("reasoning_content").is_string()) return fail("Invalid agent reasoning");
                preserve = preserve || !message.at("reasoning_content").get<std::string>().empty();
            }
            if (index++ == 0 && role == "system") {
                if (message.contains("tool_calls") || message.contains("tool_call_id")) return fail("System cannot call tools");
                continue;
            }
            if (role != expected) return fail("Agent roles or pending tool results do not match");
            if (role == "user") {
                if (message.contains("tool_calls") || message.contains("tool_call_id")) return fail("User cannot contain tool fields");
                expected = "assistant";
            } else if (role == "assistant") {
                if (message.contains("tool_call_id")) return fail("Assistant cannot be a tool result");
                if (message.contains("tool_calls")) {
                    const auto & calls = message.at("tool_calls");
                    if (!calls.is_array() || calls.empty() || calls.size() > 16) return fail("Agent tool call count");
                    for (const auto & call : calls) {
                        if (!call.is_object() || call.size() != 3 || call.value("type", "") != "function" ||
                            !call.contains("id") || !call.at("id").is_string() || !call.contains("function")) return fail("Invalid typed tool call");
                        auto id = call.at("id").get<std::string>();
                        const auto & function = call.at("function");
                        if (!named(id, 128) || !seen.insert(id).second || !function.is_object() || function.size() != 2 ||
                            !function.contains("name") || !function.at("name").is_string() ||
                            !named(function.at("name").get<std::string>(), 64) || !function.contains("arguments") ||
                            !function.at("arguments").is_string() ||
                            !common_json::parse(function.at("arguments").get<std::string>()).is_object()) return fail("Invalid function arguments or ID");
                        pending.insert(id);
                    }
                }
                expected = pending.empty() ? "user" : "tool";
            } else if (role == "tool") {
                if (message.contains("tool_calls") || !message.contains("tool_call_id") || !message.at("tool_call_id").is_string() ||
                    pending.erase(message.at("tool_call_id").get<std::string>()) != 1) return fail("Unknown or repeated tool result");
                expected = pending.empty() ? "assistant" : "tool";
            } else return fail("Unsupported agent role");
        }
        if (!pending.empty() || (last != "user" && last != "tool")) return fail("Agent needs user turn or completed tool results");
        // The installed Qwen XML template iterates argument mappings. OpenAI's
        // wire representation is a JSON string; normalize that representation
        // only after validation, without altering the argument values.
        for (auto & message : messages) if (message.contains("tool_calls"))
            for (auto & call : message.at("tool_calls")) {
                auto & arguments = call.at("function").at("arguments");
                arguments = common_json::parse(arguments.get<std::string>());
            }
        // JSON-in-JSON may contain escaped control markers. Check the decoded
        // argument values and keys as well as their wire representation.
        total = nodes = 0;
        inspect(inspect, input, 0);
        input["enable_thinking"] = enable_thinking; input["preserve_thinking"] = preserve;
        input["add_generation_prompt"] = true;
        jinja::context context(v->chat_template);
        jinja::global_from_json(context, input, false);
        jinja::runtime runtime(context);
        const auto result = jinja::runtime::gather_string_parts(runtime.execute(*v->program))->val_str.str();
        if (result.size() > 524288) return fail("Rendered agent chat exceeds 512 KiB");
        if (result.size() > size_t(capacity)) return -int32_t(result.size());
        if (!result.empty()) std::memcpy(output, result.data(), result.size());
        return int32_t(result.size());
    } catch (const std::exception & error) { return fail(error.what()); }
      catch (...) { return fail("Unknown agent rendering failure"); }
}
