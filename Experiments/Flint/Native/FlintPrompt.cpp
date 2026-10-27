// Publisher template and normalization: recursive-oss/rec 0e3150ca (Apache-2.0).
// Rendering uses pinned minja (MIT) with Python-compatible tojson separators.
#include "FlashFlint.h"
#include "FlintTemplate.h"
#include <minja/minja.hpp>
#include <mutex>
#include <cstring>
extern void flint_set_error(const std::string &);
static void replace_all(std::string & text, const std::string & from, const std::string & to) {
    for (size_t at = 0; (at = text.find(from, at)) != std::string::npos; at += to.size()) text.replace(at, from.size(), to);
}
int flf_render_prompt(const char * text, int32_t bytes, char * output, int32_t capacity) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    try {
        if (!text || bytes < 1 || bytes > (2 << 20) || capacity < 0 || (capacity && !output))
            throw std::runtime_error("Invalid Flint template input/output buffer");
        auto request = json::parse(text, text + bytes);
        auto & messages = request.at("messages");
        if (!messages.is_array() || messages.empty() || messages.size() > 4096) throw std::runtime_error("Invalid Flint messages");
        for (auto & message : messages) {
            const auto role = message.at("role").get<std::string>();
            if (role != "system" && role != "developer" && role != "user" && role != "assistant" && role != "tool")
                throw std::runtime_error("Unsupported Flint role");
            if (!message.contains("content") || message["content"].is_null()) message["content"] = "";
            auto content = message.at("content").get<std::string>();
            if (role != "assistant") {
                replace_all(content, std::string(1, '\0'), "");
                replace_all(content, "<think>", ""); replace_all(content, "</think>", "");
                for (const char * token : {"begin_of_text", "end_of_text", "system", "developer", "user", "assistant", "observation"})
                    replace_all(content, std::string("<|") + token + "|>", std::string("&lt;|") + token + "|&gt;");
                message["content"] = content;
            }
            if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
                auto & calls = message["tool_calls"];
                if (!calls.is_array() || calls.size() > 16) throw std::runtime_error("Invalid Flint tool-call history");
                for (auto & call : calls) {
                    auto & fn = call.contains("function") ? call["function"] : call;
                    auto & args = fn.at("arguments");
                    if (args.is_string()) {
                        const auto value = args.get<std::string>();
                        args = value.find_first_not_of(" \t\r\n") == std::string::npos ? json::object() : json::parse(value);
                    }
                    if (!args.is_object() || args.size() > 128) throw std::runtime_error("Flint tool arguments must be objects");
                }
            }
        }
        if (!request.contains("tools") || request["tools"].is_null()) request["tools"] = json::array();
        if (!request["tools"].is_array() || request["tools"].size() > 64) throw std::runtime_error("Invalid Flint tool definitions");
        if (!request.contains("add_generation_prompt")) request["add_generation_prompt"] = true;
        if (!request["add_generation_prompt"].is_boolean()) throw std::runtime_error("Invalid generation-prompt flag");
        static auto root = minja::Parser::parse(FLINT_CHAT_TEMPLATE, {true, true});
        auto context = minja::Context::make(minja::Value(request));
        // minja at this pin has no ensure_ascii keyword. The official template
        // requests False, so provide the compatible filter without editing it.
        context->set("tojson", minja::simple_function("tojson", {"value", "ensure_ascii", "indent"},
            [](const std::shared_ptr<minja::Context> &, minja::Value & args) -> minja::Value {
                if (args.get<bool>("ensure_ascii", false)) throw std::runtime_error("Flint requires UTF-8 JSON");
                return minja::Value(args.at("value").dump(-1, true));
            }));
        const auto rendered = root->render(context);
        if (rendered.empty() || rendered.size() > (2u << 20)) throw std::runtime_error("Flint prompt exceeds 2 MiB");
        flint_set_error("");
        if (size_t(capacity) < rendered.size()) return -int(rendered.size());
        std::memcpy(output, rendered.data(), rendered.size()); return int(rendered.size());
    } catch (const std::exception & e) { flint_set_error(e.what()); return 0; }
}
