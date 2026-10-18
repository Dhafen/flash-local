#include "FlashLlamaReference.h"
#include "../../Sources/CLlamaTokenizer/DequantizedLinearReference.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <vector>
#include <unistd.h>

static bool never_abort(void *) { return false; }
static bool always_abort(void *) { return true; }
static void check(bool value, const char * reason) {
    if (!value) throw std::runtime_error(reason);
}
using context_owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using reference_owner = std::unique_ptr<flr_context, decltype(&flr_free)>;

// Compare against an independently executed all-F32 graph made from the same
// dequantized original weights. Only tiny explicitly named random fixtures
// may be materialized this way; the phone oracle never creates this file.
static uint64_t make_f32_fixture(const char * source, const char * target) {
    check(std::filesystem::file_size(source) <= (32u << 20), "Only bounded random fixtures may be converted");
    ggml_context * tensors = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> input(gguf_init_from_file(source, {false, &tensors}), gguf_free);
    context_owner data(tensors, ggml_free);
    check(input && data, "Random fixture could not be read");
    const auto name = gguf_find_key(input.get(), "general.name");
    const auto arch = gguf_find_key(input.get(), "general.architecture");
    check(name >= 0 && arch >= 0 && gguf_get_kv_type(input.get(), name) == GGUF_TYPE_STRING &&
        gguf_get_kv_type(input.get(), arch) == GGUF_TYPE_STRING &&
        std::string(gguf_get_val_str(input.get(), name)) == "synthetic-graph-test" &&
        std::string(gguf_get_val_str(input.get(), arch)) == "qwen35moe", "Not an untrained Ornith fixture");
    const int64_t count = gguf_get_n_tensors(input.get());
    check(count > 0 && count <= 128, "Synthetic tensor count bound");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    gguf_set_kv(output.get(), input.get());
    std::vector<std::vector<float>> payload(size_t(count), std::vector<float>{});
    size_t bytes = 0;
    for (int64_t i = 0; i < count; ++i) {
        auto * t = ggml_get_tensor(data.get(), gguf_get_tensor_name(input.get(), i));
        check(t && ggml_is_contiguous(t) && t->ne[0] > 0, "Synthetic tensor shape");
        const size_t values = size_t(ggml_nelements(t));
        check(values <= (64u << 20) / sizeof(float) && bytes <= (64u << 20) - values * sizeof(float), "Synthetic F32 payload bound");
        bytes += values * sizeof(float); payload[i].resize(values);
        if (t->type == GGML_TYPE_F32) std::memcpy(payload[i].data(), t->data, values * sizeof(float));
        else {
            const auto convert = ggml_get_type_traits(t->type)->to_float;
            check(convert != nullptr, "Synthetic type has no dequantizer");
            convert(t->data, payload[i].data(), int64_t(values));
        }
        gguf_add_tensor(output.get(), t);
        gguf_set_tensor_type(output.get(), t->name, GGML_TYPE_F32);
        gguf_set_tensor_data(output.get(), t->name, payload[i].data());
    }
    check(gguf_write_to_file(output.get(), target, false), "Synthetic F32 write failed");
    const auto layers = gguf_find_key(input.get(), "qwen35moe.block_count");
    const auto mtp = gguf_find_key(input.get(), "qwen35moe.nextn_predict_layers");
    const auto topk = gguf_find_key(input.get(), "qwen35moe.expert_used_count");
    check(layers >= 0 && mtp >= 0 && topk >= 0, "Synthetic routing metadata unavailable");
    return uint64_t(gguf_get_val_u32(input.get(), layers) - gguf_get_val_u32(input.get(), mtp)) *
        gguf_get_val_u32(input.get(), topk) * 5;
}

static std::vector<std::vector<float>> replay(const char * path, bool dequantized, flr_dequantized_metrics & metrics) {
    const int fd = open(path, O_RDONLY);
    check(fd >= 0, "Synthetic descriptor unavailable");
    reference_owner model(dequantized ? flr_create_dequantized(fd, 32, never_abort, nullptr, nullptr, nullptr) :
        flr_create(fd, 32, never_abort, nullptr), flr_free);
    close(fd);
    if (!model) throw std::runtime_error(flr_last_error());
    const int size = flr_vocabulary_size(model.get());
    check(size > 0 && size <= 4096, "Synthetic vocabulary bound");
    std::vector<std::vector<float>> result;
    for (int token : {1, 2, 3, 15, 4}) {
        std::vector<float> row(size);
        if (flr_step(model.get(), token, row.data(), size) != size) throw std::runtime_error(flr_last_error());
        result.push_back(std::move(row));
    }
    metrics = flr_dequantized_snapshot(model.get());
    return result;
}

static void operator_controls() {
    ggml_cpu_init();
    double largest_rounding_gap = 0;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0,
            GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        ggml_init_params params{256 << 10, nullptr, true};
        context_owner ctx(ggml_init(params), ggml_free);
        constexpr int columns = 256, rows = 7, experts = 4, selected = 2, tokens = 3;
        auto * w = ggml_new_tensor_3d(ctx.get(), type, columns, rows, experts);
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, columns, selected, tokens);
        auto * all_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 8, tokens);
        auto * ids = ggml_view_2d(ctx.get(), all_ids, selected, tokens, all_ids->nb[1], 0);
        auto * y = ggml_mul_mat_id(ctx.get(), w, x, ids);
        auto * dense_w = ggml_view_2d(ctx.get(), w, columns, rows, w->nb[1], 0);
        auto * dense_x = ggml_view_2d(ctx.get(), x, columns, tokens, x->nb[2], 0);
        auto * dense_y = ggml_mul_mat(ctx.get(), dense_w, dense_x);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
            ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()), ggml_backend_buffer_free);
        check(bool(storage), "Operator fixture buffer allocation");
        ggml_backend_buffer_set_usage(storage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> original(columns * rows * experts), activation(columns * selected * tokens), decoded(original.size());
        for (size_t i = 0; i < original.size(); ++i) original[i] = .3f * std::cos(float(i) * .091f) + .006f * float(i % 7);
        for (size_t i = 0; i < activation.size(); ++i) activation[i] = .7f * std::sin(float(i) * .117f) + .03f * std::cos(float(i) * 1.7f);
        std::vector<uint64_t> packed((ggml_nbytes(w) + 7) / 8);
        check(ggml_quantize_chunk(type, original.data(), packed.data(), 0, rows * experts, columns, nullptr) == ggml_nbytes(w),
            "Operator canonical quantizer byte count");
        ggml_backend_tensor_set(w, packed.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, activation.data(), 0, activation.size() * sizeof(float));
        if (type == GGML_TYPE_F32) decoded = original;
        else ggml_get_type_traits(type)->to_float(packed.data(), decoded.data(), int64_t(decoded.size()));
        int32_t selections[24] = {3, 1, -1, -1, -1, -1, -1, -1, 0, 3, -1, -1, -1, -1, -1, -1, 2, 0, -1, -1, -1, -1, -1, -1};
        ggml_backend_tensor_set(all_ids, selections, 0, sizeof(selections));
        flash_local_dequantized_linear oracle;
        check(oracle.apply(y, never_abort, nullptr) && oracle.apply(dense_y, never_abort, nullptr), "Linear oracle did not select fixture");
        check(oracle.expert_calls == 1 && oracle.dense_calls == 1 && oracle.maximum_scratch_bytes <= (1u << 20), "Linear oracle metrics");
        const auto * rhs = ggml_get_type_traits_cpu(ggml_get_type_traits_cpu(type)->vec_dot_type);
        std::vector<uint64_t> rounded((ggml_row_size(ggml_get_type_traits_cpu(type)->vec_dot_type, columns) + 7) / 8);
        for (int token = 0; token < tokens; ++token) for (int rank = 0; rank < selected; ++rank) for (int row = 0; row < rows; ++row) {
            double expected = 0;
            const int expert = selections[token * 8 + rank];
            for (int col = 0; col < columns; ++col) expected +=
                double(decoded[(expert * rows + row) * columns + col]) * double(activation[(token * selected + rank) * columns + col]);
            float value;
            ggml_backend_tensor_get(y, &value, token * y->nb[2] + rank * y->nb[1] + row * 4, 4);
            check(std::abs(double(value) - expected) < 1e-6, "Strided top-k or expert-specific input not preserved");
            if (token == 0 && rank == 0) {
                rhs->from_float(activation.data(), rounded.data(), columns);
                float approximate;
                ggml_get_type_traits_cpu(type)->vec_dot(columns, &approximate, 0,
                    reinterpret_cast<const char *>(packed.data()) + (expert * rows + row) * ggml_row_size(type, columns), 0,
                    rounded.data(), 0, 1);
                largest_rounding_gap = std::max(largest_rounding_gap, std::abs(double(approximate) - expected));
            }
        }
        for (int token = 0; token < tokens; ++token) for (int row = 0; row < rows; ++row) {
            double expected = 0;
            for (int col = 0; col < columns; ++col) expected += double(decoded[row * columns + col]) * double(activation[token * selected * columns + col]);
            float value; ggml_backend_tensor_get(dense_y, &value, token * dense_y->nb[1] + row * 4, 4);
            check(std::abs(double(value) - expected) < 1e-6, "Strided dense input not preserved");
        }
        std::vector<uint8_t> before(ggml_nbytes(y)); ggml_backend_tensor_get(y, before.data(), 0, before.size());
        selections[8] = experts;
        ggml_backend_tensor_set(all_ids, selections, 0, sizeof(selections));
        bool rejected = false;
        try { oracle.apply(y, never_abort, nullptr); } catch (const std::exception &) { rejected = true; }
        std::vector<uint8_t> after(before.size()); ggml_backend_tensor_get(y, after.data(), 0, after.size());
        check(rejected && before == after, "Invalid real expert ID was not atomically rejected");
        rejected = false;
        try { oracle.apply(dense_y, always_abort, nullptr); } catch (const std::exception &) { rejected = true; }
        check(rejected, "Linear oracle ignored cancellation/governor");
    }
    check(largest_rounding_gap > 1e-4, "Independent reference did not distinguish CPU activation rounding");
    std::cout << "ORNITH_DEQUANTIZED_OPERATORS types=8 denseAndExperts=true stridedTopK=true originalF32Input=true invalidExpertAtomic=true governor=true boundedScratch=true syntheticOnly=true\n";
}

int main(int argc, char ** argv) {
    try {
        check(argc == 3, "Expected bounded synthetic source and temporary F32 target");
        operator_controls();
        const auto expected_decisions = make_f32_fixture(argv[1], argv[2]);
        flr_dequantized_metrics unused{}, metrics{}, f32_metrics{};
        const auto ordinary_before = replay(argv[1], false, unused);
        const auto expected = replay(argv[2], false, unused);
        const auto exact = replay(argv[1], true, metrics);
        (void)replay(argv[2], true, f32_metrics);
        const auto ordinary_after = replay(argv[1], false, unused);
        check(ordinary_before == ordinary_after, "Third oracle altered the ordinary CPU reference");
        check(metrics.dense_calls > 0 && metrics.expert_calls >= 60 && metrics.rows > 0 &&
            metrics.maximum_scratch_bytes > 0 && metrics.maximum_scratch_bytes <= (1u << 20), "Full graph oracle has no linear coverage");
        check(metrics.routing_decisions == expected_decisions && metrics.routing_decisions == f32_metrics.routing_decisions &&
            metrics.canonical_routing_digest == f32_metrics.canonical_routing_digest,
            "Independent dequantized graph routing differs or is incomplete");
        double maximum = 0, squared = 0, norm = 0;
        check(expected.size() == exact.size(), "Independent graph step count");
        for (size_t step = 0; step < exact.size(); ++step) {
            check(expected[step].size() == exact[step].size(), "Independent vocabulary shape");
            for (size_t token = 0; token < exact[step].size(); ++token) {
                const double a = exact[step][token], b = expected[step][token], delta = a - b;
                check(std::isfinite(a) && std::isfinite(b), "Nonfinite independent graph output");
                maximum = std::max(maximum, std::abs(delta)); squared += delta * delta; norm += b * b;
            }
        }
        const double relative = std::sqrt(squared / std::max(norm, 1e-30));
        check(maximum <= .0001 && relative <= .0001, "Independent all-F32 dequantized graph mismatch");
        std::cout << "ORNITH_DEQUANTIZED_REFERENCE independentF32Graph=true originalCPUUnchanged=true steps=5 fullVocabulary=true"
            << " denseCalls=" << metrics.dense_calls << " expertCalls=" << metrics.expert_calls
            << " maxAbs=" << maximum << " relativeL2=" << relative << " syntheticOnly=true physicalPhone=false\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
