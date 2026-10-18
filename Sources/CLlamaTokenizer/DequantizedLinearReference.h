#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <memory>

// Diagnostic third oracle, never a production kernel. The scheduler pauses
// after each selected model-weight multiplication; replace that result with
// original dequantized weights dot the original F32 activation, accumulated
// in double. Subsequent graph operations consume this result. No model-wide
// conversion, activation quantization, weight mutation or global type override.
// This is deliberately distinct from the unchanged pinned CPU vec_dot path.
struct flash_local_dequantized_linear {
    uint64_t dense_calls = 0, expert_calls = 0, rows = 0, maximum_scratch_bytes = 0;
    static constexpr int64_t maximum_columns = 65536, maximum_rows = 300000;
    struct alignas(64) scratch_buffers {
        std::array<float, maximum_columns> input, decoded;
        std::array<uint64_t, maximum_columns / 2> packed;
    };
    static_assert(sizeof(scratch_buffers) <= (1u << 20), "Linear scratch budget");
    std::unique_ptr<scratch_buffers> scratch;

    static void require(bool value, const char * reason) {
        if (!value) throw std::runtime_error(reason);
    }
    static ggml_backend_buffer_t buffer(const ggml_tensor * t) {
        for (int i = 0; t && i < 8; ++i, t = t->view_src) if (t->buffer) return t->buffer;
        return nullptr;
    }
    static bool selected(const ggml_tensor * t) {
        if (!t || (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) || !t->src[0]) return false;
        const auto * w = t->src[0];
        // Quantized operands cannot be an attention/KV matrix. F16/BF16/F32
        // model parameters are identified by their owner's WEIGHTS usage;
        // runtime attention products keep their original implementation.
        const auto b = buffer(w);
        return ggml_is_quantized(w->type) ||
            (b && ggml_backend_buffer_get_usage(b) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    static void host(const ggml_tensor * t) {
        const auto b = buffer(t);
        require(t && t->data && b && ggml_backend_buft_is_host(ggml_backend_buffer_get_type(b)),
            "Dequantized reference requires CPU-accessible tensors");
    }
    bool apply(ggml_tensor * dst, bool (*abort)(void *), void * user) {
        if (!selected(dst)) return false;
        const auto * w = dst->src[0]; const auto * x = dst->src[1];
        host(w); host(x); host(dst);
        require(abort && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            w->ne[0] > 0 && w->ne[0] <= maximum_columns && w->ne[1] > 0 && w->ne[1] <= maximum_rows &&
            w->ne[3] == 1 && x->ne[3] == 1 && dst->ne[3] == 1 &&
            x->ne[0] == w->ne[0] && dst->ne[0] == w->ne[1] &&
            w->nb[0] == ggml_type_size(w->type) && x->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
            w->ne[0] % ggml_blck_size(w->type) == 0,
            "Dequantized reference linear geometry/type bound");
        const auto to_float = ggml_get_type_traits(w->type)->to_float;
        require(to_float || w->type == GGML_TYPE_F32, "Dequantized reference weight conversion unsupported");
        const size_t row_bytes = ggml_row_size(w->type, w->ne[0]);
        require(w->nb[1] >= row_bytes && row_bytes <= size_t(maximum_columns) * sizeof(float),
            "Dequantized reference packed row bound");
        const auto * ids = dst->op == GGML_OP_MUL_MAT_ID ? dst->src[2] : nullptr;
        if (ids) {
            host(ids);
            require(ids->type == GGML_TYPE_I32 && ids->ne[0] > 0 && ids->ne[0] <= 8 &&
                ids->ne[1] > 0 && ids->ne[1] <= 128 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
                ids->nb[0] == sizeof(int32_t) && ids->nb[1] >= size_t(ids->ne[0]) * sizeof(int32_t) &&
                w->ne[2] > 0 && w->ne[2] <= 256 && x->ne[1] > 0 && ids->ne[0] % x->ne[1] == 0 &&
                x->ne[2] == ids->ne[1] && dst->ne[1] == ids->ne[0] && dst->ne[2] == ids->ne[1],
                "Dequantized reference selected-expert geometry bound");
            // Validate every real expert selection before any output write.
            for (int64_t token = 0; token < ids->ne[1]; ++token) for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                int32_t expert;
                std::memcpy(&expert, static_cast<const char *>(ids->data) + token * ids->nb[1] + rank * ids->nb[0], sizeof(expert));
                require(expert >= 0 && expert < w->ne[2], "Dequantized reference expert ID out of range");
            }
        } else {
            require(w->ne[2] == 1 && x->ne[1] > 0 && x->ne[1] <= 128 && x->ne[2] > 0 && x->ne[2] <= 128 &&
                dst->ne[1] == x->ne[1] && dst->ne[2] == x->ne[2],
                "Dequantized reference dense broadcast bound");
        }
        const size_t columns = size_t(w->ne[0]);
        // A single lazy, fixed allocation avoids vector-growth transients and
        // keeps the default reference at zero additional linear scratch.
        if (!scratch) scratch = std::make_unique<scratch_buffers>();
        maximum_scratch_bytes = sizeof(scratch_buffers);
        auto & input = scratch->input; auto & decoded = scratch->decoded; auto & packed = scratch->packed;
        for (int64_t token = 0; token < dst->ne[2]; ++token) for (int64_t rank = 0; rank < dst->ne[1]; ++rank) {
            int32_t expert = 0;
            if (ids) std::memcpy(&expert, static_cast<const char *>(ids->data) + token * ids->nb[1] + rank * ids->nb[0], sizeof(expert));
            const size_t input_offset = token * x->nb[2] + (rank % x->ne[1]) * x->nb[1];
            std::memcpy(input.data(), static_cast<const char *>(x->data) + input_offset, columns * sizeof(float));
            require(std::all_of(input.begin(), input.begin() + columns, [](float a) { return std::isfinite(a); }),
                "Dequantized reference input nonfinite");
            for (int64_t row = 0; row < w->ne[1]; ++row) {
                if (row % 64 == 0) require(!abort(user), "Dequantized reference governor stopped calculation");
                const size_t offset = size_t(expert) * w->nb[2] + size_t(row) * w->nb[1];
                std::memcpy(packed.data(), static_cast<const char *>(w->data) + offset, row_bytes);
                if (w->type == GGML_TYPE_F32) std::memcpy(decoded.data(), packed.data(), columns * sizeof(float));
                else to_float(packed.data(), decoded.data(), int64_t(columns));
                double sum = 0;
                for (size_t column = 0; column < columns; ++column) sum += double(decoded[column]) * double(input[column]);
                const float result = float(sum);
                require(std::isfinite(result), "Dequantized reference output nonfinite");
                std::memcpy(static_cast<char *>(dst->data) + token * dst->nb[2] + rank * dst->nb[1] + row * sizeof(float),
                    &result, sizeof(result));
                rows++;
            }
        }
        if (ids) expert_calls++; else dense_calls++;
        return true;
    }
};
