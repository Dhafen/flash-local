// Untrained operator controls, never a trained-model or iPhone speed claim.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#import <Foundation/Foundation.h>

extern "C" void flash_omni_metal_snapshot(uint64_t *, uint64_t *, uint64_t *, uint64_t *);
extern "C" void flash_omni_memory_snapshot(uint64_t *, uint64_t *, uint64_t *, uint64_t *, bool *);
using json = nlohmann::json;
using context_owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_owner = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
static constexpr double absolute_limit = 1e-4, relative_limit = 1e-4;
static void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
static float sample(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return float(int32_t(state >> 8) - 8388608) / 8388608.f;
}
struct fixture {
    ggml_type type;
    int columns, rows, experts, selected, input_ranks;
    size_t row_bytes, bytes;
    std::vector<uint64_t> packed;
    std::vector<float> input;
    std::vector<int32_t> ids;
    fixture(ggml_type type, int columns, int rows, bool routed, bool shared, float amplitude) :
        type(type), columns(columns), rows(rows), experts(routed ? 256 : 1),
        selected(routed ? 8 : 1), input_ranks(routed && !shared ? 8 : 1),
        row_bytes(ggml_row_size(type, columns)), bytes(row_bytes * rows * experts),
        packed((bytes + 7) / 8), input(size_t(columns) * input_ranks * 4), ids(256 * 4, -1) {
        check(bytes <= (8u << 20) && columns <= 2048 && rows <= 64, "Synthetic payload bound");
        std::vector<float> source(columns);
        uint32_t state = 0x12345678u;
        for (int row = 0; row < rows * experts; ++row) {
            for (float & value : source) value = amplitude * sample(state);
            auto * destination = reinterpret_cast<char *>(packed.data()) + row * row_bytes;
            check(ggml_quantize_chunk(type, source.data(), destination, 0, 1, columns, nullptr) == row_bytes,
                  "Canonical synthetic quantizer byte count");
        }
        state = 0x87654321u;
        for (float & value : input) value = sample(state);
        const int choices[] = {255, 0, 117, 31, 7, 123, 65, 8};
        for (int token = 0; token < 4; ++token) for (int rank = 0; rank < selected; ++rank)
            ids[token * 256 + rank] = routed ? choices[(rank + token) % 8] : 0;
    }
    std::vector<float> reference(int batch) const {
        std::vector<float> result(size_t(rows) * selected * batch), row(columns);
        const auto decode = ggml_get_type_traits(type)->to_float;
        check(decode != nullptr, "Missing original-weight row dequantizer");
        for (int token = 0; token < batch; ++token) for (int rank = 0; rank < selected; ++rank) {
            const int expert = ids[token * 256 + rank], input_rank = input_ranks == 1 ? 0 : rank;
            const float * activation = input.data() + size_t(token * input_ranks + input_rank) * columns;
            for (int output = 0; output < rows; ++output) {
                decode(reinterpret_cast<const char *>(packed.data()) + size_t(expert * rows + output) * row_bytes,
                       row.data(), columns);
                double sum = 0;
                for (int column = 0; column < columns; ++column) sum += double(row[column]) * double(activation[column]);
                result[size_t(token * selected + rank) * rows + output] = float(sum);
            }
        }
        return result;
    }
};
struct errors {
    double maximum = 0, squared = 0, denominator = 0;
    double relative() const { return std::sqrt(squared / std::max(denominator, 1e-30)); }
    bool passed() const { return maximum <= absolute_limit && relative() <= relative_limit; }
};
static errors compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    check(actual.size() == expected.size() && !actual.empty(), "Projection output shape");
    errors out;
    for (size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]), "Non-finite projection output");
        const double d = double(actual[i]) - double(expected[i]);
        out.maximum = std::max(out.maximum, std::abs(d)); out.squared += d * d;
        out.denominator += double(expected[i]) * double(expected[i]);
    }
    return out;
}
static std::vector<float> metal(const fixture & f, int batch) {
    std::vector<float> result(size_t(f.rows) * f.selected * batch);
    @autoreleasepool {
        backend_owner backend(ggml_backend_metal_init(), ggml_backend_free);
        check(backend && ggml_backend_is_metal(backend.get()), "Actual Metal backend is required");
        const auto device = ggml_backend_get_device(backend.get());
        check(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU, "CPU fallback is forbidden");
        context_owner context(ggml_init({2u << 20, nullptr, true}), ggml_free);
        check(bool(context), "Synthetic graph metadata allocation");
        auto * weights = f.experts == 1
            ? ggml_new_tensor_2d(context.get(), f.type, f.columns, f.rows)
            : ggml_new_tensor_3d(context.get(), f.type, f.columns, f.rows, f.experts);
        ggml_set_name(weights, "synthetic.weight");
        auto * input = f.experts == 1
            ? ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, f.columns, batch)
            : ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, f.columns, f.input_ranks, batch);
        ggml_tensor * all_ids = nullptr, * output = nullptr;
        if (f.experts == 1) output = ggml_mul_mat(context.get(), weights, input);
        else {
            all_ids = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 256, batch);
            auto * ids = ggml_view_2d(context.get(), all_ids, f.selected, batch, all_ids->nb[1], 0);
            output = ggml_mul_mat_id(context.get(), weights, input, ids);
        }
        check(ggml_backend_supports_op(backend.get(), output), "Metal operator is unsupported");
        auto * graph = ggml_new_graph_custom(context.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        buffer_owner storage(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
        check(bool(storage), "Synthetic Metal tensor allocation");
        ggml_backend_tensor_set(weights, f.packed.data(), 0, f.bytes);
        ggml_backend_tensor_set(input, f.input.data(), 0, ggml_nbytes(input));
        if (all_ids) ggml_backend_tensor_set(all_ids, f.ids.data(), 0, ggml_nbytes(all_ids));
        check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "Metal graph computation failed");
        ggml_backend_synchronize(backend.get());
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    }
    uint64_t live = 0, owned = 0, mapped = 0, allocated = 0; bool observed = false;
    flash_omni_memory_snapshot(&live, &owned, &mapped, &allocated, &observed);
    check(observed && live == 0 && owned == 0 && mapped == 0, "Metal operator buffers survived their owner");
    return result;
}
int main(int argc, char ** argv) {
    json report{{"syntheticOnly", true}, {"physicalPhone", false}, {"trainedWeightPayloadPresent", false},
        {"trainedModelNumericsVerified", false}, {"phonePerformanceMeasured", false},
        {"referenceArithmetic", "dequantized-weight-f32-double-v1"},
        {"absoluteLimit", absolute_limit}, {"relativeL2Limit", relative_limit},
        {"maximumPackedMatrixBytes", 0}, {"comparisons", json::array()}};
    bool completed = false, passed = true;
    try {
        check(argc == 2, "Expected aggregate JSON output path");
        ggml_cpu_init(); // Canonical test-weight quantization, never projection fallback.
        @autoreleasepool {
            for (float amplitude : {0.004f, 0.04f, 0.4f}) for (int kind = 0; kind < 4; ++kind) {
                const bool routed = kind >= 2, shared = kind == 2;
                const ggml_type type = kind % 2 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
                const int columns = kind == 3 ? 512 : 2048;
                fixture data(type, columns, routed ? 8 : 64, routed, shared, amplitude);
                report["maximumPackedMatrixBytes"] = std::max(report["maximumPackedMatrixBytes"].get<size_t>(), data.bytes);
                auto single = metal(data, 1), corrupted = single;
                corrupted[0] += 1;
                check(!compare(corrupted, data.reference(1)).passed(), "Changed projection output was accepted");
                for (int batch : {1, 4}) {
                    const auto output = batch == 1 ? single : metal(data, batch);
                    const auto error = compare(output, data.reference(batch));
                    std::vector<float> first(output.begin(), output.begin() + single.size());
                    passed = passed && error.passed();
                    report["comparisons"].push_back({{"type", ggml_type_name(type)}, {"columns", columns},
                        {"rows", data.rows}, {"experts", data.experts}, {"selected", data.selected},
                        {"inputRanks", data.input_ranks}, {"batch", batch}, {"sourceAmplitude", amplitude},
                        {"maximumAbsoluteError", error.maximum}, {"relativeL2Error", error.relative()},
                        {"serialFirstTokenMaximumAbsoluteError", compare(first, single).maximum}, {"withinLimits", error.passed()}});
                }
            }
        }
        uint64_t dispatches = 0, buffers = 0, gpu_ns = 0, timestamped = 0;
        flash_omni_metal_snapshot(&dispatches, &buffers, &gpu_ns, &timestamped);
        check(dispatches >= 24 && buffers >= 24, "Actual native Metal dispatch evidence is missing");
        report["metalDispatches"] = dispatches; report["metalCommandBuffers"] = buffers;
        report["metalTimestampedCommandBuffers"] = timestamped; report["gpuNanoseconds"] = gpu_ns;
        report["metalExecutionVerified"] = true; report["allOwnedBuffersReleased"] = true;
        report["corruptedOutputRejected"] = true; completed = true;
    } catch (const std::exception & error) {
        std::cerr << "Metal projection control: " << error.what() << '\n'; passed = false;
    }
    report["probeCompleted"] = completed; report["projectionLimitsPassed"] = completed && passed;
    if (argc == 2) { std::ofstream output(argv[1]); output << report.dump(2) << '\n'; }
    std::cout << "ORNITH_METAL_PROJECTIONS completed=" << (completed ? "true" : "false")
              << " limitsPassed=" << (completed && passed ? "true" : "false") << " syntheticOnly=true physicalPhone=false\n";
    return completed && passed ? 0 : 1;
}
