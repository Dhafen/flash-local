#include "FlashOrnithLES.h"
#include "VerifiedDescriptor.h"
#include "LoaderDiagnostics.h"
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
namespace fs = std::filesystem;
using json = nlohmann::json;
struct output { std::vector<std::vector<float>> logits; flo_metrics metrics; };
static output evaluate(int fd, const char * dir, bool metal, int depth, int steps, int cache = 8) {
    flo_options options{};
    options.context_tokens = steps > 5 ? 256 : 32; options.prefill_slots = 16; options.cache_slots_per_layer = cache;
    options.lookahead_depth = depth; options.io_threads = 2; options.cpu_threads = 2;
    options.metal = metal; options.trace_routing = true; options.memory_ceiling_bytes = 5500000000ULL;
    flo_context * context = flo_open(fd, dir, options, nullptr, nullptr);
    if (!context) throw std::runtime_error(flo_last_error());
    if (flo_snapshot(nullptr).live_contexts != 1) throw std::runtime_error("Missing live native context ownership");
    output out;
    try {
        const int32_t tokens[] = {1, 2, 3, 15, 4};
        for (int index = 0; index < steps; ++index) {
            const int32_t token = tokens[index % 5];
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
    flo_close(context);
    const auto released = flo_snapshot(nullptr);
    if (released.live_contexts || released.live_metal_buffers || released.owned_metal_bytes || released.mapped_metal_bytes)
        throw std::runtime_error("Native context or buffer ownership survived close");
    return out;
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
        const std::string capability = "/dev/fd/" + std::to_string(fd);
        std::error_code descriptor_error;
        if (flash_local_expert_file_size(capability, descriptor_error) != fs::file_size(argv[1]) || descriptor_error)
            throw std::runtime_error("Verified descriptor length changed");
        const int retained = flash_local_open_experts(capability);
        if (retained < 0 || retained == fd) throw std::runtime_error("Verified descriptor was not retained");
        close(retained);
        for (const char * invalid : {"/dev/fd/", "/dev/fd/-1", "/dev/fd/+1", "/dev/fd/1suffix"}) {
            bool rejected = false;
            try { (void) flash_local_verified_descriptor(invalid); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("Malformed descriptor capability was accepted");
        }
        flash_local_loader_diagnostics diagnostic;
        diagnostic.begin();
        diagnostic.record(GGML_LOG_LEVEL_INFO, "must not be exported");
        diagnostic.record(GGML_LOG_LEVEL_ERROR, "tensor 'x' unavailable in /private/container/model.gguf\n");
        std::string message = diagnostic.describe("load failed");
        if (message.find("tensor 'x'") == std::string::npos || message.find("/private") != std::string::npos ||
            message.find("must not be exported") != std::string::npos)
            throw std::runtime_error("Native loader error privacy regression");
        diagnostic.end(); diagnostic.begin();
        if (diagnostic.describe("next load") != "next load") throw std::runtime_error("Stale native loader error");
        diagnostic.record(GGML_LOG_LEVEL_ERROR, std::string(8192, 'x').c_str());
        if (diagnostic.describe("load").size() > 2054) throw std::runtime_error("Unbounded native loader error");
        diagnostic.end();
        if (flo_prepare(fd, dir.c_str(), nullptr, nullptr)) throw std::runtime_error(flo_last_error());
        const fs::path invalid = dir / "invalid-header";
        fs::create_directory(invalid);
        fs::copy_file(dir / "manifest-template.json", invalid / "manifest-template.json");
        { std::ofstream broken(invalid / "dense.gguf", std::ios::binary); broken << "BAD!"; }
        bool native_cause_exported = false;
        try { evaluate(fd, invalid.c_str(), false, 0, 1); }
        catch (const std::runtime_error & error) {
            const std::string reason = error.what();
            native_cause_exported = reason.rfind("LES model loading failed: ", 0) == 0 &&
                reason.size() <= 2200 && reason.find(invalid.string()) == std::string::npos;
        }
        if (!native_cause_exported) throw std::runtime_error("Missing bounded native loader cause");
        const int steps = fs::path(argv[1]).filename().string().find("top8") == std::string::npos ? 5 : 128;
        std::ifstream manifest_input(dir / "manifest-template.json");
        json manifest; manifest_input >> manifest;
        const int bad_depth = std::min(17, manifest.at("model").at("expert_count").get<int>() + 1);
        bool invalid_depth_rejected = false;
        try { evaluate(fd, dir.c_str(), false, bad_depth, 1); }
        catch (const std::runtime_error & error) {
            const std::string reason = error.what();
            invalid_depth_rejected = reason == "LES lookahead exceeds expert count" || reason == "Invalid scheduling options";
        }
        if (!invalid_depth_rejected) throw std::runtime_error("Invalid lookahead was not rejected before graph construction");
        const auto cpu = evaluate(fd, nullptr, false, 0, steps);
        const auto streamed = evaluate(fd, dir.c_str(), false, 0, steps);
        const auto ahead = evaluate(fd, dir.c_str(), false, 2, steps);
        const auto page = evaluate(fd, dir.c_str(), false, 0, steps, 0);
        const double error = std::max({compare(cpu, streamed), compare(cpu, ahead), compare(cpu, page)});
        if (error > 1e-6 || !ahead.metrics.io_calls || !ahead.metrics.io_bytes)
            throw std::runtime_error("LES correctness/instrumentation gate failed");
        json report{{"syntheticOnly", true}, {"physicalPhone", false}, {"revision", flo_revision()},
            {"residentCPUReference", true}, {"cpuStreamingMaximumAbsoluteError", error},
            {"metalExecutionVerified", false}, {"actualRoutingIdentical", true},
            {"invalidLookaheadRejected", true},
            {"verifiedDescriptorMetadata", true}, {"boundedPrivateLoaderErrors", true},
            {"nativeLoaderCauseExported", true}, {"nativeOwnersReleasedAfterEachContext", true},
            {"resetBitwiseIdentical", true}, {"sameGreedy", true}, {"modelRequantized", false},
            {"gpuTimestampsAvailable", false},
            {"expertPayloadCopied", false}, {"positions", steps}, {"cpuLogits", cpu.logits}};
        std::ofstream out(argv[3]); out << report.dump(2) << '\n'; out.close(); close(fd);
        std::cout << "ORNITH_LES_CONTROL cpuMaxAbs=" << error << " metalExecutionVerified=false"
                  << " routing=true reset=true noRequantization=true noExpertCopy=true syntheticOnly=true\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
