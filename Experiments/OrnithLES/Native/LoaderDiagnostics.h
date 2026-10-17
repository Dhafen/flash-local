#pragma once
#include "ggml.h"
#include <cstring>
#include <mutex>
#include <regex>
#include <string>

// Only errors during native model/context loading enter the exported failure.
// Retain the first useful cause; the loader's final generic message is not enough.
class flash_local_loader_diagnostics {
    std::mutex mutex;
    bool active = false;
    std::string errors;
public:
    void begin() { std::lock_guard<std::mutex> lock(mutex); errors.clear(); active = true; }
    void end() { std::lock_guard<std::mutex> lock(mutex); active = false; }
    void record(ggml_log_level level, const char * text) noexcept {
        if (level != GGML_LOG_LEVEL_ERROR || !text) return;
        try {
            // Do not export app-container paths or descriptor numbers.
            std::string safe(text, strnlen(text, 2048));
            safe = std::regex_replace(safe, std::regex(R"(/[^\s]+)"), "[local-path]");
            for (char & c : safe) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            std::lock_guard<std::mutex> lock(mutex);
            if (active && errors.size() < 2048) errors.append(safe.substr(0, 2048 - errors.size()));
        } catch (...) { /* A diagnostic must never unwind through a C callback. */ }
    }
    std::string describe(const char * fallback) {
        std::lock_guard<std::mutex> lock(mutex);
        return errors.empty() ? std::string(fallback) : std::string(fallback) + ": " + errors;
    }
};
