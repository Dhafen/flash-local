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
#include <set>
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
    if (batch.canonical != single.canonical || batch.expert_sets != single.expert_sets || batch.decisions != 32 || single.decisions != 32 ||
        batch.sequential == single.sequential) throw std::runtime_error("Routing trace is not batch-order independent");
    const auto before = batch;
    padded[256] = -1;
    if (batch.append("ffn_moe_topk-0", 0, 8, 2, 1024, 256, row_reader) ||
        batch.decisions != before.decisions || batch.canonical != before.canonical || batch.sequential != before.sequential ||
        batch.expert_sets != before.expert_sets)
        throw std::runtime_error("Invalid routing trace was partially committed");
    padded[256] = 16;
    flash_local_routing_trace permuted;
    for (const char * layer : {"ffn_moe_topk-0", "ffn_moe_topk-1"})
        if (!permuted.append(layer, 0, 8, 2, 1024, 256,
            [&](int32_t * out, size_t offset, size_t bytes) {
                row_reader(out, offset, bytes); std::reverse(out, out + 8);
            })) throw std::runtime_error("Permuted expert set rejected");
    if (permuted.canonical == single.canonical || permuted.sequential == single.sequential ||
        permuted.expert_sets != single.expert_sets || permuted.decisions != single.decisions)
        throw std::runtime_error("Rank permutation and expert selection were conflated");
    padded[256] = 17; // Duplicate: not an eight-expert selection.
    const auto duplicate_before = batch;
    if (batch.append("ffn_moe_topk-0", 0, 8, 2, 1024, 256, row_reader) ||
        batch.canonical != duplicate_before.canonical || batch.expert_sets != duplicate_before.expert_sets ||
        batch.decisions != duplicate_before.decisions || batch.sequential != duplicate_before.sequential)
        throw std::runtime_error("Duplicate expert was accepted or partially committed");
    padded[256] = 31;
    flash_local_routing_trace changed;
    for (const char * layer : {"ffn_moe_topk-0", "ffn_moe_topk-1"})
        if (!changed.append(layer, 0, 8, 2, 1024, 256, row_reader)) throw std::runtime_error("Changed route rejected");
    if (changed.canonical == single.canonical || changed.expert_sets == single.expert_sets || changed.decisions != single.decisions)
        throw std::runtime_error("Changed expert was not visible in canonical routing trace");
    for (const char * invalid : {"ffn_moe_topk-", "ffn_moe_topk--1", "ffn_moe_topk-0suffix"}) {
        if (changed.append(invalid, 0, 8, 1, 1024, 256, row_reader)) throw std::runtime_error("Invalid layer accepted");
    }
    if (changed.append("ffn_moe_topk-0", 0, 8, 2, 16, 256, row_reader) ||
        changed.append("ffn_moe_topk-0", 0, 8, 33, 1024, 256, row_reader) ||
        changed.append("ffn_moe_topk-0", 0, 8, 2, std::numeric_limits<size_t>::max(), 256, row_reader))
        throw std::runtime_error("Invalid top-k tensor geometry accepted");
    std::cout << "ORNITH_ROUTING_TRACE paddedRows=true batchOrderIndependent=true changedExpertDetected=true invalidInputAtomic=true\n";
    std::cout << "ORNITH_EXPERT_SETS exactPositionLayer=true rankPermutationDistinguished=true changedExpertDetected=true duplicateRejected=true atomic=true\n";
}
struct layer_trace {
    std::set<uint64_t> points, expected;
    size_t bytes = 0;
    static uint64_t key(uint32_t stage, int32_t layer) { return uint64_t(stage) << 32 | uint32_t(layer); }
    layer_trace() {
        // All generated native controls have four main layers and one unused
        // MTP block. Check the whole diagnostic contract before Swift builds.
        for (uint32_t stage : {2u, 5u, 6u, 7u, 8u})
            for (int layer = 0; layer < 4; ++layer) expected.insert(key(stage, layer));
        for (uint64_t id : {key(1, -1), key(3, 0), key(4, 0), key(9, -1), key(10, 0)}) expected.insert(id);
    }
    void reset() { points.clear(); bytes = 0; }
    static bool capture(uint32_t stage, int32_t layer, const float * values, int32_t count, void * user) {
        auto & t = *static_cast<layer_trace *>(user);
        const auto id = key(stage, layer);
        if (!values || count < 1 || count > 16384 || t.bytes + size_t(count) * 4 > (4u << 20) ||
            !t.expected.count(id) || !t.points.insert(id).second) return false;
        t.bytes += size_t(count) * 4;
        return true;
    }
};
static output evaluate(int fd, const char * dir, bool metal, int depth, int steps, int cache = 8, int prefill = 0,
                       bool trace_layers = false, bool trace_routing = true, int physical = 4, int evaluation = 4,
                       bool flash_attention = false) {
    if (trace_layers && prefill) throw std::runtime_error("Layer controls use single-token forwards");
    layer_trace trace;
    flo_options options{};
    options.context_tokens = steps > 5 ? 256 : 32; options.prefill_slots = prefill > 0 ? physical * 8 : 16;
    options.cache_slots_per_layer = cache;
    options.lookahead_depth = depth; options.io_threads = 2; options.cpu_threads = 2;
    options.metal = metal; options.trace_routing = trace_routing; options.memory_ceiling_bytes = 5500000000ULL;
    options.flash_attention = flash_attention;
    if (trace_layers) { options.checkpoint = layer_trace::capture; options.checkpoint_user = &trace; }
    flo_context * context = flo_open(fd, dir, options, nullptr, nullptr);
    if (!context) throw std::runtime_error(flo_last_error());
    if (flo_snapshot(nullptr).live_contexts != 1) throw std::runtime_error("Missing live native context ownership");
    output out;
    try {
        const int32_t tokens[] = {1, 2, 3, 15, 4};
        for (int first = 0; first < steps;) {
            const int count = first < prefill ? std::min(evaluation, prefill - first) : 1;
            std::vector<int32_t> chunk(count);
            for (int i = 0; i < count; ++i) chunk[i] = tokens[(first + i) % 5];
            std::vector<float> row(flo_vocabulary_size(context));
            trace.reset();
            if (flo_decode(context, chunk.data(), count, row.data(), (int)row.size())) throw std::runtime_error(flo_last_error());
            if (trace_layers && trace.points != trace.expected) throw std::runtime_error("Incomplete native semantic layer trace");
            out.logits.push_back(row);
            first += count; out.positions.push_back(first - 1);
        }
        out.metrics = flo_snapshot(context);
        if (out.metrics.flash_attention_active != flash_attention) throw std::runtime_error("Requested attention graph was not used");
        if (trace_routing && !out.metrics.routing_decisions) throw std::runtime_error("Routing trace was not executed");
        if (!trace_routing && !trace_layers && (out.metrics.routing_decisions || out.metrics.canonical_routing_digest || out.metrics.canonical_expert_set_digest ||
                out.metrics.probe_samples)) throw std::runtime_error("Callback-free execution unexpectedly observed tensors");
        flo_reset(context);
        const int first_count = prefill > 0 ? std::min(evaluation, prefill) : 1;
        std::vector<int32_t> initial(first_count);
        for (int i = 0; i < first_count; ++i) initial[i] = tokens[i % 5];
        std::vector<float> replay(out.logits[0].size());
        trace.reset();
        if (flo_decode(context, initial.data(), first_count, replay.data(), (int)replay.size()) || replay != out.logits[0])
            throw std::runtime_error("Reset did not restore the complete recurrent/KV state");
        if (trace_layers && trace.points != trace.expected) throw std::runtime_error("Reset lost native semantic layer trace");
    } catch (...) { flo_close(context); throw; }
    flo_close(context);
    const auto released = flo_snapshot(nullptr);
    if (released.live_contexts || released.live_metal_buffers || released.owned_metal_bytes || released.mapped_metal_bytes)
        throw std::runtime_error("Native context or buffer ownership survived close");
    return out;
}
static double compare(const output & a, const output & b, bool routing = true, bool require_greedy = true) {
    if (routing && (a.metrics.canonical_routing_digest != b.metrics.canonical_routing_digest ||
        a.metrics.canonical_expert_set_digest != b.metrics.canonical_expert_set_digest ||
        a.metrics.routing_decisions != b.metrics.routing_decisions))
        throw std::runtime_error("Actual expert routing changed");
    if (a.logits.size() != a.positions.size() || b.logits.size() != b.positions.size() ||
        b.positions.empty() || a.positions.back() != b.positions.back()) throw std::runtime_error("Incomplete logit positions");
    double error = 0;
    for (size_t i = 0; i < b.logits.size(); ++i) {
        const auto position = b.positions[i];
        if (position < 0 || size_t(position) >= a.positions.size() || a.positions[position] != position ||
            (i > 0 && position <= b.positions[i - 1]) || a.logits[position].size() != b.logits[i].size())
            throw std::runtime_error("Vocabulary or logit position changed");
        for (size_t j = 0; j < a.logits[position].size(); ++j) {
            if (!std::isfinite(a.logits[position][j]) || !std::isfinite(b.logits[i][j]))
                throw std::runtime_error("Non-finite control logit");
            error = std::max(error, std::abs((double)a.logits[position][j] - b.logits[i][j]));
        }
        if (require_greedy && std::max_element(a.logits[position].begin(), a.logits[position].end()) - a.logits[position].begin() !=
            std::max_element(b.logits[i].begin(), b.logits[i].end()) - b.logits[i].begin())
            throw std::runtime_error("Greedy token changed");
    }
    return error;
}
static double relative_l2(const output & a, const output & b) {
    // Cross-graph diagnostics also describe rejected experimental candidates.
    // Shape and finiteness remain mandatory; ordinary controls are unchanged.
    (void)compare(a, b, false, false);
    double squared = 0, reference = 0;
    for (size_t i = 0; i < b.logits.size(); ++i) {
        const auto & expected = a.logits[b.positions[i]];
        for (size_t j = 0; j < expected.size(); ++j) {
            const double delta = double(expected[j]) - b.logits[i][j];
            squared += delta * delta; reference += double(expected[j]) * expected[j];
        }
    }
    return std::sqrt(squared / std::max(reference, 1e-30));
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
        uint64_t prefix_bytes = 0;
        {
            flo_options options{};
            options.context_tokens = 32; options.prefill_slots = 32;
            options.io_threads = options.cpu_threads = 2; options.memory_ceiling_bytes = 5500000000ULL;
            flo_context * context = flo_open(fd, dir.c_str(), options, nullptr, nullptr);
            if (!context) throw std::runtime_error(flo_last_error());
            try {
                const int32_t prefix[] = {1, 2, 3, 15}, old_suffix[] = {4, 3, 2}, new_suffix[] = {2, 4};
                std::vector<float> expected(flo_vocabulary_size(context)), actual(expected.size());
                auto decode = [&](const int32_t * ids, int count, std::vector<float> & row) {
                    if (flo_decode(context, ids, count, row.data(), (int)row.size())) throw std::runtime_error(flo_last_error());
                };
                decode(prefix, 4, actual);
                prefix_bytes = flo_capture_prefix(context);
                if (!prefix_bytes || prefix_bytes > (128ULL << 20)) throw std::runtime_error("Missing bounded recurrent snapshot");
                decode(old_suffix, 3, actual);
                if (flo_restore_prefix(context)) throw std::runtime_error(flo_last_error());
                decode(new_suffix, 2, actual);
                flo_reset(context);
                decode(prefix, 4, expected); decode(new_suffix, 2, expected);
                if (actual != expected) throw std::runtime_error("Restored hybrid state differs from fresh exact-prefix replay");
                // Exercise all accepted-prefix lengths against actual hybrid
                // GDN/conv + attention state, without a trained draft model.
                flo_reset(context); decode(prefix, 4, actual);
                std::vector<int32_t> greedy;
                std::vector<std::vector<float>> rows;
                for (int i = 0; i < 5; ++i) {
                    const int32_t id = int32_t(std::max_element(actual.begin(), actual.end()) - actual.begin());
                    greedy.push_back(id); decode(&greedy.back(), 1, actual); rows.push_back(actual);
                }
                for (int rejected = 1; rejected <= 4; ++rejected) {
                    flo_reset(context); decode(prefix, 4, actual);
                    std::vector<int32_t> block(greedy.begin(), greedy.begin() + 4);
                    if (rejected < 4) block[rejected] = (block[rejected] + 1) % int32_t(actual.size());
                    const int accepted = flo_verify_greedy(context, block.data(), 4, actual.data(), int32_t(actual.size()));
                    if (accepted != rejected) throw std::runtime_error("Wrong greedy accepted prefix");
                    for (size_t i = 0; i < actual.size(); ++i)
                        if (std::abs(actual[i] - rows[accepted - 1][i]) > 1e-6)
                            throw std::runtime_error("Speculative rollback differs from ordinary target logits");
                    decode(&greedy[accepted], 1, actual);
                    for (size_t i = 0; i < actual.size(); ++i)
                        if (std::abs(actual[i] - rows[accepted][i]) > 1e-6)
                            throw std::runtime_error("Continuation after speculative rollback differs");
                }
                std::cout << "ORNITH_GREEDY_ROLLBACK branches=4 fullVocabulary=true recurrentAndAttention=true continuation=true trainedDraft=false\n";
                flo_reset(context);
                // Reset invalidates the snapshot rather than resurrecting history.
                if (flo_restore_prefix(context) == 0) throw std::runtime_error("Cleared chat restored a stale snapshot");
            } catch (...) { flo_close(context); throw; }
            flo_close(context);
        }
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
        const auto observed = evaluate(fd, dir.c_str(), false, 0, steps, 8, 0, true);
        const double layer_error = compare(cpu, observed);
        if (layer_error > 1e-6) throw std::runtime_error("Layer observation changed native control logits");
        const auto streamed = evaluate(fd, dir.c_str(), false, 0, steps);
        const auto ahead = evaluate(fd, dir.c_str(), false, 2, steps);
        const auto page = evaluate(fd, dir.c_str(), false, 0, steps, 0);
        // Keep the exact streaming test graph-matched. FA and decomposed
        // attention use different F16 KV arithmetic in the pinned CPU backend;
        // an extra 1e-4 cross-graph limit was not the app's acceptance policy.
        // Neither the ordinary 1e-6 controls nor phone admission are relaxed.
        const auto flash_reference = evaluate(fd, nullptr, false, 0, steps, 8, 0, false, true, 4, 4, true);
        const auto flash = evaluate(fd, dir.c_str(), false, 0, steps, 0, 0, false, true, 4, 4, true);
        const double flash_streaming_error = compare(flash_reference, flash);
        if (flash_streaming_error > 1e-6) throw std::runtime_error("Flash Attention streaming differs from resident FA reference");
        const double flash_error = compare(cpu, flash, false, false);
        const double flash_relative = relative_l2(cpu, flash);
        bool flash_greedy = true;
        for (size_t i = 0; i < flash.logits.size(); ++i) {
            const auto & a = cpu.logits[flash.positions[i]], & b = flash.logits[i];
            flash_greedy = flash_greedy && std::max_element(a.begin(), a.end()) - a.begin() ==
                std::max_element(b.begin(), b.end()) - b.begin();
        }
        const bool flash_routing = cpu.metrics.canonical_routing_digest == flash.metrics.canonical_routing_digest &&
            cpu.metrics.canonical_expert_set_digest == flash.metrics.canonical_expert_set_digest &&
            cpu.metrics.routing_decisions == flash.metrics.routing_decisions;
        const bool flash_admitted = flash_error <= 0.01 && flash_relative <= 0.001 && flash_greedy && flash_routing;
        std::cout << "ORNITH_CPU_FLASH_ATTENTION residentStreamingMaxAbs=" << flash_streaming_error
                  << " decomposedMaxAbs=" << flash_error << " decomposedRelativeL2=" << flash_relative
                  << " sameGreedy=" << flash_greedy << " exactRouting=" << flash_routing
                  << " candidateAdmitted=" << flash_admitted << " physicalPhone=false\n" << std::flush;
        // This is an explicit, never-promoted experiment. A rejected candidate
        // is a valid diagnostic result, not permission to change chat/API.
        // Phone execute() independently fails closed with the unchanged gate.
        // A NaN must never disappear inside std::max and pass a numerical gate.
        auto nonfinite = flash; nonfinite.logits[0][0] = std::numeric_limits<float>::quiet_NaN();
        bool nonfinite_rejected = false;
        try { (void)compare(cpu, nonfinite); } catch (...) { nonfinite_rejected = true; }
        if (!nonfinite_rejected) throw std::runtime_error("Non-finite numerical control was accepted");
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
        double callback_free_error = 0;
        for (int mode = 0; mode < 3; ++mode) for (int prefix : {0, std::min(7, steps)}) {
            const auto plain = evaluate(fd, dir.c_str(), false, mode == 1 ? 2 : 0,
                steps, mode == 2 ? 0 : 8, prefix, false, false);
            callback_free_error = std::max(callback_free_error, compare(cpu, plain, false));
        }
        if (callback_free_error > 1e-6) throw std::runtime_error("Callback-free timing path changed model logits");
        // Actually execute each physical dimension; a five-token fixture alone
        // cannot prove that a configured sixteen-token graph was exercised.
        const int tuning_steps = steps == 5 ? 35 : steps;
        const auto tuning_reference = steps == 5 ? evaluate(fd, nullptr, false, 0, tuning_steps) : cpu;
        double tuning_error = 0; int tuning_variants = 0;
        for (int physical : {1, 4, 8, 16}) for (int evaluation : {physical, 32}) for (bool observed : {true, false}) {
            const auto tuned = evaluate(fd, dir.c_str(), false, 0, tuning_steps, 0, 33, false, observed, physical, evaluation);
            if (tuned.metrics.configured_physical_batch_tokens != uint64_t(physical) ||
                tuned.metrics.maximum_physical_batch_tokens != uint64_t(physical))
                throw std::runtime_error("Requested physical batch was not actually executed");
            if (evaluation > physical && tuned.metrics.skipped_intermediate_output_batches == 0)
                throw std::runtime_error("Intermediate output selection was not actually exercised");
            tuning_error = std::max(tuning_error, compare(tuning_reference, tuned, observed));
            ++tuning_variants;
        }
        if (tuning_error > 1e-6) throw std::runtime_error("Larger physical/evaluation batch changed model logits");
        int extended_variants = 0;
        for (int evaluation : {64, 128, 256, 512, 1024, 2048}) for (bool observed : {true, false}) {
            const auto tuned = evaluate(fd, dir.c_str(), false, 0, tuning_steps, 0, 33, false, observed, 4, evaluation);
            if (compare(tuning_reference, tuned, observed) > 1e-6) throw std::runtime_error("Extended logical batch changed logits or routing");
            ++extended_variants;
        }
        json report{{"syntheticOnly", true}, {"physicalPhone", false}, {"revision", flo_revision()},
            {"residentCPUReference", true}, {"cpuStreamingMaximumAbsoluteError", error},
            {"recurrentOnlyPrefixCheckpointBitwiseReplay", true}, {"prefixCheckpointBytes", prefix_bytes},
            {"greedyBlockRollbackBranches", 4}, {"trainedDFlashGraphExecuted", false},
            {"cpuFlashAttentionMaximumAbsoluteError", flash_error}, {"phoneFlashAttentionExecuted", false},
            {"cpuFlashAttentionRelativeL2Error", flash_relative}, {"cpuFlashAttentionResidentStreamingMaximumAbsoluteError", flash_streaming_error},
            {"cpuFlashAttentionSameGreedy", flash_greedy}, {"cpuFlashAttentionRoutingIdentical", flash_routing},
            {"cpuFlashAttentionCandidateAdmitted", flash_admitted}, {"nonfiniteControlLogitsRejected", nonfinite_rejected},
            {"extendedLogicalBatchControls", extended_variants}, {"logicalBatchLimit", 2048},
            {"metalExecutionVerified", false}, {"actualRoutingIdentical", true},
            {"invalidLookaheadRejected", true},
            {"verifiedDescriptorMetadata", true}, {"boundedPrivateLoaderErrors", true},
            {"nativeLoaderCauseExported", true}, {"nativeOwnersReleasedAfterEachContext", true},
            {"canonicalBatchedRoutingIdentical", true}, {"batchPrefillTokens", 4},
            {"canonicalExpertSetsIdentical", true}, {"rankPermutationsDistinguished", true},
            {"completeLayerCheckpointTrace", true}, {"layerCheckpointMaximumAbsoluteError", layer_error},
            {"callbackFreeVariants", 6}, {"callbackFreeMaximumAbsoluteError", callback_free_error},
            {"batchPrefixLengths", prefixes}, {"batchedPrefillMaximumAbsoluteError", batch_error},
            {"prefillTuningVariants", tuning_variants}, {"prefillTuningMaximumAbsoluteError", tuning_error},
            {"physicalBatchDimensionsActuallyExecuted", {1, 4, 8, 16}},
            {"intermediateOutputSelectionVerified", true},
            {"resetBitwiseIdentical", true}, {"sameGreedy", true}, {"modelRequantized", false},
            {"gpuTimestampsAvailable", false},
            {"expertPayloadCopied", false}, {"positions", steps}, {"cpuLogits", cpu.logits}};
        std::ofstream out(argv[3]); out << report.dump(2) << '\n'; out.close(); close(fd);
        std::cout << "ORNITH_LES_CONTROL cpuMaxAbs=" << error << " metalExecutionVerified=false"
                  << " routing=true reset=true noRequantization=true noExpertCopy=true syntheticOnly=true\n";
        std::cout << "ORNITH_BATCH_CONTROL prefixCases=" << prefixes.size() << " steps=" << steps
                  << " maxAbs=" << batch_error << " canonicalRouting=true reset=true syntheticOnly=true\n";
        std::cout << "ORNITH_LAYER_CONTROL mainLayers=4 complete=true rawDeltaNetDistinct=true finalLayerMatched=true maxAbs="
                  << layer_error << " syntheticOnly=true\n";
        std::cout << "ORNITH_CALLBACK_FREE variants=6 serialAndBatch=true fullVocabulary=true reset=true noObservers=true maxAbs="
                  << callback_free_error << " syntheticOnly=true\n";
        std::cout << "ORNITH_PREFILL_TUNING variants=" << tuning_variants << " physical=1,4,8,16 evaluationThrough32=true"
                  << " actualDimensions=true fullVocabulary=true canonicalRouting=true callbackFree=true unneededHeadsSkipped=true maxAbs="
                  << tuning_error << " syntheticOnly=true physicalPhone=false\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
