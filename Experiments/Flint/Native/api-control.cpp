#include "FlashFlint.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

static void require(bool ok, const char * text) { if (!ok) throw std::runtime_error(text); }
int main(int argc, char ** argv) {
    flf_context * c = nullptr;
    try {
        if (argc != 3) throw std::runtime_error("small-flint.gguf empty-private-directory");
        require(std::filesystem::file_size(argv[1]) < (64u << 20), "Control admits small random fixtures only");
        std::filesystem::create_directories(argv[2]);
        const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
        require(fd >= 0, "Cannot open fixture");
        flf_options o{256, 32, 4, 2, 5500000000ULL, 400000000ULL, false, true, true, false};
        auto rejected = o; rejected.process_ceiling_bytes = 1024;
        require(flf_open(fd, argv[2], rejected, nullptr, nullptr, nullptr) == nullptr, "Over-budget load accepted");
        rejected = o; rejected.metal = true; rejected.diagnostic_routing = false;
        require(flf_open(fd, argv[2], rejected, nullptr, nullptr, nullptr) == nullptr, "Non-diagnostic Metal F32 KV accepted");
        rejected = o; rejected.metal = true; rejected.f32_kv_control = false;
        rejected.diagnostic_routing = false; rejected.diagnostic_disable_flash = true;
        require(flf_open(fd, argv[2], rejected, nullptr, nullptr, nullptr) == nullptr, "Non-diagnostic Flash override accepted");
        rejected = o; rejected.diagnostic_disable_flash = true;
        require(flf_open(fd, argv[2], rejected, nullptr, nullptr, nullptr) == nullptr, "CPU Flash override accepted");
        require(std::filesystem::is_empty(argv[2]), "Rejected load left a manifest");
        c = flf_open(fd, argv[2], o, nullptr, nullptr, nullptr); close(fd);
        if (!c) throw std::runtime_error(flf_last_error());
        // A second model must not replace/free the first model's ownership.
        require(flf_open(-1, argv[2], o, nullptr, nullptr, nullptr) == nullptr, "Duplicate load accepted");
        require(flf_vocab_size(c) == 32, "Wrong fixture vocabulary");
        require(flf_library_path() && std::filesystem::is_regular_file(flf_library_path()), "Loaded native binary identity is missing");
        require(flf_sample(c) < 0, "Sampling accepted missing logits/configuration");
        require(flf_configure_sampling(c, std::numeric_limits<float>::quiet_NaN(), 1, 0, 42) != 0, "NaN sampling accepted");
        require(flf_configure_sampling(c, 1, 0, 0, 42) != 0, "Zero top-p accepted");
        require(flf_configure_sampling(c, 1, 1, 33, 42) != 0, "Invalid vocabulary top-k accepted");
        std::vector<float> control(32), state_only(32);
        const int32_t prefix[] = {1, 2, 3, 4};
        require(flf_evaluate(c, prefix, 4, control.data(), 32) == 0, flf_last_error());
        const int32_t last[] = {5};
        require(flf_evaluate(c, last, 1, control.data(), 32) == 0, flf_last_error());
        require(flf_snapshot(c).cached_tokens == 5, "Append KV position was lost");
        require(flf_reset(c) == 0, "Reset failed");
        require(flf_evaluate(c, prefix, 4, nullptr, 0) == 0, flf_last_error());
        require(flf_evaluate(c, last, 1, state_only.data(), 32) == 0, flf_last_error());
        for (int i = 0; i < 32; ++i) require(std::isfinite(control[i]) && std::abs(control[i] - state_only[i]) < 1e-6f, "State-only prefill changed logits");
        require(flf_snapshot(c).state_only_tokens == 4 && flf_snapshot(c).cached_tokens == 5 && flf_snapshot(c).io_bytes > 0, "Missing real prefill/KV/I/O statistics");
        require(flf_reset(c) == 0 && flf_sample(c) < 0, "Reset kept stale logits");
        require(flf_configure_sampling(c, 0, 1, 0, 42) == 0, flf_last_error());
        require(flf_evaluate(c, prefix, 4, nullptr, 0) == 0 && flf_sample(c) < 0, "State-only prefill exposed stale logits");
        require(flf_evaluate_for_sampling(c, last, 1) == 0, flf_last_error());
        require(flf_copy_logits(c, state_only.data(), 31) < 0, "Short diagnostic logit buffer accepted");
        require(flf_copy_logits(c, state_only.data(), 32) == 32, flf_last_error());
        for (int i = 0; i < 32; ++i) require(control[i] == state_only[i], "No-copy evaluation changed logits");
        require(flf_sample(c) == std::max_element(control.begin(), control.end()) - control.begin(), "Native greedy sampling changed top-1");
        std::vector<int32_t> sequence;
        for (int pass = 0; pass < 2; ++pass) {
            require(flf_reset(c) == 0 && flf_configure_sampling(c, 1, .95f, 20, 42) == 0, flf_last_error());
            require(flf_evaluate_for_sampling(c, prefix, 4) == 0, flf_last_error());
            for (int step = 0; step < 24; ++step) {
                const int32_t token = flf_sample(c);
                require(token >= 0 && token < 32, "Native sampler returned invalid ID");
                if (pass == 0) sequence.push_back(token);
                else require(sequence[step] == token, "Seeded decoding is not reproducible");
                require(flf_evaluate_for_sampling(c, &token, 1) == 0, flf_last_error());
            }
        }
        require(flf_reset(c) == 0 && flf_trace_routes(c, true) == 0, flf_last_error());
        require(flf_evaluate(c, prefix, 4, control.data(), 32) == 0, flf_last_error());
        const int needed = flf_routing_json(c, nullptr, 0);
        require(needed < 0 && needed > -(4 << 20), "Invalid routing diagnostic size");
        std::vector<char> raw(-needed);
        require(flf_routing_json(c, raw.data(), raw.size()) == -needed, flf_last_error());
        const auto routes = nlohmann::json::parse(raw.begin(), raw.end());
        std::set<std::pair<int,int>> visited;
        for (const auto & row : routes) {
            const int position = row.at("position"), layer = row.at("layer");
            require(position >= 0 && position < 4 && layer >= 2 && layer < 6, "Routing positions/layers differ from actual prefill");
            require(visited.insert({position, layer}).second && row.at("experts").size() == 6, "Duplicate/incomplete routing decision");
        }
        // Final-layer FFNs for unrequested output positions are pruned. K/V
        // still advances for every token; only position 3 needs final logits.
        require(visited.size() == 13 && visited.count({3, 5}) == 1 && visited.count({0, 5}) == 0,
            "Routing diagnostics did not match state-only final-layer pruning");
        require(flf_reset(c) == 0 && flf_trace_routes(c, true) == 0, flf_last_error());
        const int32_t eight[] = {1, 2, 3, 4, 5, 6, 7, 8};
        require(flf_evaluate(c, eight, 8, control.data(), 32) == 0, flf_last_error());
        const int more = flf_routing_json(c, nullptr, 0); require(more < 0, "Missing multi-microbatch routing trace");
        raw.resize(-more); require(flf_routing_json(c, raw.data(), raw.size()) == -more, flf_last_error());
        const auto split = nlohmann::json::parse(raw.begin(), raw.end());
        visited.clear();
        for (const auto & row : split) require(visited.emplace(row.at("position").get<int>(), row.at("layer").get<int>()).second,
            "Duplicate microbatch position");
        require(visited.size() == 25 && visited.count({7, 5}) == 1 && visited.count({4, 2}) == 1,
            "Physical microbatch offsets were not retained in routing diagnostics");
        require(flf_trace_routes(c, false) == 0 && flf_routing_json(c, nullptr, 0) == 0, "Disabled diagnostics leaked a stale trace");
        flf_cancel(c);
        require(flf_evaluate(c, last, 1, state_only.data(), 32) != 0, "Cancellation ignored");
        require(flf_reset(c) == 0 && flf_snapshot(c).cached_tokens == 0, "Cancellation recovery did not reset KV");
        require(flf_evaluate(c, prefix, 4, control.data(), 32) == 0, flf_last_error());
        const uint64_t f32_kv_bytes = flf_snapshot(c).estimated_kv_bytes;
        flf_close(c); c = nullptr;
        require(std::filesystem::is_empty(argv[2]), "Private manifest not cleaned on unload");
        const int f16_fd = open(argv[1], O_RDONLY | O_CLOEXEC);
        require(f16_fd >= 0, "Cannot reopen fixture for F16 KV diagnostic");
        o.f32_kv_control = false;
        c = flf_open(f16_fd, argv[2], o, nullptr, nullptr, nullptr); close(f16_fd);
        if (!c) throw std::runtime_error(flf_last_error());
        require(flf_snapshot(c).estimated_kv_bytes * 2 == f32_kv_bytes,
            "F32 and F16 KV budgets do not match their element sizes");
        require(flf_evaluate(c, prefix, 4, state_only.data(), 32) == 0, flf_last_error());
        for (float value : state_only) require(std::isfinite(value), "F16 KV diagnostic produced nonfinite logits");
        flf_close(c); c = nullptr;
        require(std::filesystem::is_empty(argv[2]), "F16 diagnostic manifest not cleaned on unload");
        std::cout << "FLINT_API_CONTROL memoryAdmission=true duplicateLoadRejected=true descriptorRetained=true stateOnlyPrefill=true noCopyLogits=true greedy=true seededDecode=true staleLogitsRejected=true routingTrace=true kvAppend=true reset=true cancellation=true diagnosticGuards=true cpuKVF16=true cpuKVF32=true unload=true\n";
        return 0;
    } catch (const std::exception & e) {
        if (c) flf_close(c);
        std::cerr << "Flint API control: " << e.what() << std::endl; return 1;
    }
}
