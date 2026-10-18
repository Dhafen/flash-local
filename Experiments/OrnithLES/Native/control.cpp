#include "FlashOrnithLES.h"
#include "VerifiedDescriptor.h"
#include "LoaderDiagnostics.h"
#include "RoutingTrace.h"
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
#include <array>
#include <cstring>
namespace fs = std::filesystem;
using json = nlohmann::json;
struct output { std::vector<std::vector<float>> logits; std::vector<int> positions; flo_metrics metrics; };
static void routing_trace_control() {
    std::array<int32_t, 512> padded;
    padded.fill(-1);
    for (int rank = 0; rank < 8; ++rank) { padded[rank] = rank; padded[256 + rank] = 16 + rank; }
    auto row_reader = [&](int32_t * out, size_t offset, size_t bytes) {
        if ((offset != 0 && offset != 1024) || bytes != 32) throw std::runtime_error("Wrong top-k row span");
        std::memcpy(out, reinterpret_cast<const char *>(padded.data()) + offset, bytes);
    };
    flash_local_routing_trace batch, single;
    for (const char * layer : {"ffn_moe_topk-0", "ffn_moe_topk-1"})
        if (!batch.append(layer, 0, 8, 2, 1024, 256, row_reader)) throw std::runtime_error("Padded top-k trace rejected");
    for (int position = 0; position < 2; ++position) {
        for (const char * layer : {"ffn_moe_topk-0", "ffn_moe_topk-1"}) {
            if (!single.append(layer, position, 8, 1, 1024, 256,
                [&](int32_t * out, size_t, size_t bytes) { row_reader(out, size_t(position) * 1024, bytes); }))
                throw std::runtime_error("Single-position trace rejected");
        }
    }
    if (batch.canonical != single.canonical || batch.decisions != 32 || single.decisions != 32 ||
        batch.sequential == single.sequential) throw std::runtime_error("Routing trace is not batch-order independent");
    const auto before = batch;
    padded[256] = -1;
    if (batch.append("ffn_moe_topk-0", 0, 8, 2, 1024, 256, row_reader) ||
        batch.decisions != before.decisions || batch.canonical != before.canonical || batch.sequential != before.sequential)
        throw std::runtime_error("Invalid routing trace was partially committed");
    padded[256] = 17;
    flash_local_routing_trace changed;
    for (const char * layer : {"ffn_moe_topk-0", "ffn_moe_topk-1"})
        if (!changed.append(layer, 0, 8, 2, 1024, 256, row_reader)) throw std::runtime_error("Changed route rejected");
    if (changed.canonical == single.canonical || changed.decisions != single.decisions)
        throw std::runtime_error("Changed expert was not visible in canonical routing trace");
    for (const char * invalid : {"ffn_moe_topk-", "ffn_moe_topk--1", "ffn_moe_topk-0suffix"}) {
        if (changed.append(invalid, 0, 8, 1, 1024, 256, row_reader)) throw std::runtime_error("Invalid layer accepted");
    }
    if (changed.append("ffn_moe_topk-0", 0, 8, 2, 16, 256, row_reader) ||
        changed.append("ffn_moe_topk-0", 0, 8, 33, 1024, 256, row_reader) ||
        changed.append("ffn_moe_topk-0", 0, 8, 2, std::numeric_limits<size_t>::max(), 256, row_reader))
        throw std::runtime_error("Invalid top-k tensor geometry accepted");
    std::cout << "ORNITH_ROUTING_TRACE paddedRows=true batchOrderIndependent=true changedExpertDetected=true invalidInputAtomic=true\n";
}
static output evaluate(int fd, const char * dir, bool metal, int depth, int steps, int cache = 8, int prefill = 0) {
    flo_options options{};
    options.context_tokens = steps > 5 ? 256 : 32; options.prefill_slots = prefill > 0 ? 32 : 16;
    options.cache_slots_per_layer = cache;
    options.lookahead_depth = depth; options.io_threads = 2; options.cpu_threads = 2;
    options.metal = metal; options.trace_routing = true; options.memory_ceiling_bytes = 5500000000ULL;
    flo_context * context = flo_open(fd, dir, options, nullptr, nullptr);
    if (!context) throw std::runtime_error(flo_last_error());
    if (flo_snapshot(nullptr).live_contexts != 1) throw std::runtime_error("Missing live native context ownership");
    output out;
    try {
        const int32_t tokens[] = {1, 2, 3, 15, 4};
        for (int first = 0; first < steps;) {
            const int count = first < prefill ? std::min(4, prefill - first) : 1;
            int32_t chunk[4]{};
            for (int i = 0; i < count; ++i) chunk[i] = tokens[(first + i) % 5];
            std::vector<float> row(flo_vocabulary_size(context));
            if (flo_decode(context, chunk, count, row.data(), (int)row.size())) throw std::runtime_error(flo_last_error());
            out.logits.push_back(row);
            first += count; out.positions.push_back(first - 1);
        }
        out.metrics = flo_snapshot(context);
        if (!out.metrics.routing_decisions) throw std::runtime_error("Routing trace was not executed");
        flo_reset(context);
        const int first_count = prefill > 0 ? std::min(4, prefill) : 1;
        std::vector<float> replay(out.logits[0].size());
        if (flo_decode(context, tokens, first_count, replay.data(), (int)replay.size()) || replay != out.logits[0])
            throw std::runtime_error("Reset did not restore the complete recurrent/KV state");
    } catch (...) { flo_close(context); throw; }
    flo_close(context);
    const auto released = flo_snapshot(nullptr);
    if (released.live_contexts || released.live_metal_buffers || released.owned_metal_bytes || released.mapped_metal_bytes)
        throw std::runtime_error("Native context or buffer ownership survived close");
    return out;
}
static double compare(const output & a, const output & b) {
    if (a.metrics.canonical_routing_digest != b.metrics.canonical_routing_digest ||
        a.metrics.routing_decisions != b.metrics.routing_decisions)
        throw std::runtime_error("Actual expert routing changed");
    if (a.logits.size() != a.positions.size() || b.logits.size() != b.positions.size() ||
        b.positions.empty() || a.positions.back() != b.positions.back()) throw std::runtime_error("Incomplete logit positions");
    double error = 0;
    for (size_t i = 0; i < b.logits.size(); ++i) {
        const auto position = b.positions[i];
        if (position < 0 || size_t(position) >= a.positions.size() || a.positions[position] != position ||
            (i > 0 && position <= b.positions[i - 1]) || a.logits[position].size() != b.logits[i].size())
            throw std::runtime_error("Vocabulary or logit position changed");
        for (size_t j = 0; j < a.logits[position].size(); ++j)
            error = std::max(error, std::abs((double)a.logits[position][j] - b.logits[i][j]));
        if (std::max_element(a.logits[position].begin(), a.logits[position].end()) - a.logits[position].begin() !=
            std::max_element(b.logits[i].begin(), b.logits[i].end()) - b.logits[i].begin())
            throw std::runtime_error("Greedy token changed");
    }
    return error;
}
int main(int argc, char ** argv) {
    try {
        if (argc != 4 || fs::file_size(argv[1]) > (32 << 20) ||
            fs::path(argv[1]).filename().string().rfind("synthetic-ornith", 0) != 0) return 2;
        routing_trace_control();
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
        double batch_error = 0; std::vector<int> prefixes;
        for (int prefix : {1, 2, 3, 4, 5, 7, 16, 31, 32, 33, 127}) {
            if (prefix > steps) continue;
            prefixes.push_back(prefix);
            for (int mode = 0; mode < 3; ++mode) {
                const auto batched = evaluate(fd, dir.c_str(), false, mode == 1 ? 2 : 0,
                    steps, mode == 2 ? 0 : 8, prefix);
                batch_error = std::max(batch_error, compare(cpu, batched));
            }
        }
        if (batch_error > 1e-6) throw std::runtime_error("Batched prefill or decode handoff changed model logits");
        json report{{"syntheticOnly", true}, {"physicalPhone", false}, {"revision", flo_revision()},
            {"residentCPUReference", true}, {"cpuStreamingMaximumAbsoluteError", error},
            {"metalExecutionVerified", false}, {"actualRoutingIdentical", true},
            {"invalidLookaheadRejected", true},
            {"verifiedDescriptorMetadata", true}, {"boundedPrivateLoaderErrors", true},
            {"nativeLoaderCauseExported", true}, {"nativeOwnersReleasedAfterEachContext", true},
            {"canonicalBatchedRoutingIdentical", true}, {"batchPrefillTokens", 4},
            {"batchPrefixLengths", prefixes}, {"batchedPrefillMaximumAbsoluteError", batch_error},
            {"resetBitwiseIdentical", true}, {"sameGreedy", true}, {"modelRequantized", false},
            {"gpuTimestampsAvailable", false},
            {"expertPayloadCopied", false}, {"positions", steps}, {"cpuLogits", cpu.logits}};
        std::ofstream out(argv[3]); out << report.dump(2) << '\n'; out.close(); close(fd);
        std::cout << "ORNITH_LES_CONTROL cpuMaxAbs=" << error << " metalExecutionVerified=false"
                  << " routing=true reset=true noRequantization=true noExpertCopy=true syntheticOnly=true\n";
        std::cout << "ORNITH_BATCH_CONTROL prefixCases=" << prefixes.size() << " steps=" << steps
                  << " maxAbs=" << batch_error << " canonicalRouting=true reset=true syntheticOnly=true\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
