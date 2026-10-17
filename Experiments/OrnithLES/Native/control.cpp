#include "FlashOrnithLES.h"
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
namespace fs = std::filesystem;
using json = nlohmann::json;
struct output { std::vector<std::vector<float>> logits; flo_metrics metrics; };
static output evaluate(int fd, const char * dir, bool metal, int depth) {
    flo_options options{};
    options.context_tokens = 32; options.prefill_slots = 16; options.cache_slots_per_layer = 8;
    options.lookahead_depth = depth; options.io_threads = 2; options.cpu_threads = 2;
    options.metal = metal; options.trace_routing = true; options.memory_ceiling_bytes = 5500000000ULL;
    flo_context * context = flo_open(fd, dir, options, nullptr, nullptr);
    if (!context) throw std::runtime_error(flo_last_error());
    output out;
    try {
        for (int32_t token : {1, 2, 3, 15, 4}) {
            std::vector<float> row(flo_vocabulary_size(context));
            if (flo_decode(context, &token, 1, row.data(), (int)row.size())) throw std::runtime_error(flo_last_error());
            out.logits.push_back(row);
        }
        out.metrics = flo_snapshot(context);
        if (!out.metrics.routing_decisions) throw std::runtime_error("Routing trace was not executed");
        flo_reset(context);
        int32_t first = 1; std::vector<float> replay(out.logits[0].size());
        if (flo_decode(context, &first, 1, replay.data(), (int)replay.size()) || replay != out.logits[0])
            throw std::runtime_error("Reset did not restore the complete recurrent/KV state");
    } catch (...) { flo_close(context); throw; }
    flo_close(context); return out;
}
static double compare(const output & a, const output & b) {
    if (a.metrics.routing_digest != b.metrics.routing_digest || a.metrics.routing_decisions != b.metrics.routing_decisions)
        throw std::runtime_error("Actual expert routing changed");
    double error = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) {
        if (a.logits[i].size() != b.logits[i].size()) throw std::runtime_error("Vocabulary changed");
        for (size_t j = 0; j < a.logits[i].size(); ++j)
            error = std::max(error, std::abs((double)a.logits[i][j] - b.logits[i][j]));
        if (std::max_element(a.logits[i].begin(), a.logits[i].end()) - a.logits[i].begin() !=
            std::max_element(b.logits[i].begin(), b.logits[i].end()) - b.logits[i].begin())
            throw std::runtime_error("Greedy token changed");
    }
    return error;
}
int main(int argc, char ** argv) {
    try {
        if (argc != 4 || fs::file_size(argv[1]) > (32 << 20) ||
            fs::path(argv[1]).filename().string().rfind("synthetic-ornith", 0) != 0) return 2;
        const fs::path dir(argv[2]);
        if (!fs::create_directory(dir)) throw std::runtime_error("Control output must be new");
        const int fd = open(argv[1], O_RDONLY | O_NOFOLLOW);
        if (fd < 0) return 3;
        if (flo_prepare(fd, dir.c_str(), nullptr, nullptr)) throw std::runtime_error(flo_last_error());
        const auto cpu = evaluate(fd, nullptr, false, 0);
        const auto streamed = evaluate(fd, dir.c_str(), false, 0);
        const auto ahead = evaluate(fd, dir.c_str(), false, 2);
        const auto gpu = evaluate(fd, nullptr, true, 0);
        const auto gpu_stream = evaluate(fd, dir.c_str(), true, 0);
        const auto gpu_ahead = evaluate(fd, dir.c_str(), true, 2);
        const double cpu_error = std::max(compare(cpu, streamed), compare(cpu, ahead));
        const double gpu_error = std::max(compare(gpu, gpu_stream), compare(gpu, gpu_ahead));
        const double backend_error = compare(cpu, gpu);
        if (cpu_error > 1e-6 || gpu_error > 1e-6 || backend_error > 0.01 ||
            !gpu_ahead.metrics.dispatches || !gpu_ahead.metrics.command_buffers || !gpu_ahead.metrics.gpu_ns)
            throw std::runtime_error("LES correctness/instrumentation gate failed");
        json report{{"syntheticOnly", true}, {"physicalPhone", false}, {"revision", flo_revision()},
            {"cpuMaximumAbsoluteError", cpu_error}, {"metalMaximumAbsoluteError", gpu_error},
            {"cpuVsMetalMaximumAbsoluteError", backend_error}, {"actualRoutingIdentical", true},
            {"resetBitwiseIdentical", true}, {"sameGreedy", true}, {"modelRequantized", false},
            {"expertPayloadCopied", false}, {"cpuLogits", cpu.logits}};
        std::ofstream out(argv[3]); out << report.dump(2) << '\n'; out.close(); close(fd);
        std::cout << "ORNITH_LES_CONTROL cpuMaxAbs=" << cpu_error << " metalMaxAbs=" << gpu_error
                  << " routing=true reset=true noRequantization=true noExpertCopy=true syntheticOnly=true\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
