#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// A sampled, matched-input operator diagnostic, not a model reference or an
// acceptance gate. Original packed bytes + actual F32 input are read without
// modifying any tensor. Compare the observed output and the ordinary CPU
// operand path independently with a dequantized-weight double-dot reference.
struct flash_local_projection_probe {
    uint64_t samples = 0, columns = 0, weight_type = 0, cpu_input_type = 0;
    double backend_max = 0, cpu_max = 0, backend_squared = 0, cpu_squared = 0, reference_squared = 0;
    bool run(const ggml_tensor * value) {
        const auto * op = value;
        for (int depth = 0; op && op->op != GGML_OP_MUL_MAT && depth < 8; ++depth) {
            if (op->op != GGML_OP_RESHAPE && op->op != GGML_OP_VIEW && op->op != GGML_OP_CONT) return false;
            op = op->src[0];
        }
        if (!op || op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32) return false;
        const auto * w = op->src[0]; const auto * x = op->src[1];
        if (!w || !x || x->type != GGML_TYPE_F32 || w->ne[0] < 1 || w->ne[0] > 65536 ||
            w->ne[1] < 1 || w->ne[1] > 65536 || w->ne[2] != 1 || w->ne[3] != 1 ||
            x->ne[0] != w->ne[0] || ggml_nelements(x) != w->ne[0] || !ggml_is_contiguous(x) ||
            ggml_nelements(value) != w->ne[1] || !ggml_is_contiguous(value) ||
            w->ne[0] % ggml_blck_size(w->type)) return false;
        const auto * cpu = ggml_get_type_traits_cpu(w->type);
        const auto * weights = ggml_get_type_traits(w->type);
        if (!cpu->vec_dot || (!weights->to_float && w->type != GGML_TYPE_F32)) return false;
        const auto * rhs = ggml_get_type_traits_cpu(cpu->vec_dot_type);
        if (!rhs->from_float || w->ne[0] % ggml_blck_size(cpu->vec_dot_type)) return false;
        const auto row_bytes = ggml_row_size(w->type, w->ne[0]);
        const auto input_bytes = ggml_row_size(cpu->vec_dot_type, w->ne[0]);
        if (w->nb[0] != ggml_type_size(w->type) || w->nb[1] < row_bytes ||
            row_bytes > (512 << 10) || input_bytes > (512 << 10)) return false;
        std::vector<float> input(size_t(w->ne[0])), dequantized(size_t(w->ne[0]));
        std::vector<uint64_t> packed((input_bytes + 7) / 8), row((row_bytes + 7) / 8);
        ggml_backend_tensor_get(x, input.data(), 0, input.size() * sizeof(float));
        if (!std::all_of(input.begin(), input.end(), [](float f) { return std::isfinite(f); })) return false;
        rhs->from_float(input.data(), packed.data(), input.size());
        flash_local_projection_probe result;
        result.columns = w->ne[0]; result.weight_type = w->type; result.cpu_input_type = cpu->vec_dot_type;
        const int n = int(std::min<int64_t>(16, w->ne[1]));
        for (int sample = 0; sample < n; ++sample) {
            const size_t index = n == 1 ? 0 : size_t(sample) * size_t(w->ne[1] - 1) / size_t(n - 1);
            ggml_backend_tensor_get(w, row.data(), index * w->nb[1], row_bytes);
            if (w->type == GGML_TYPE_F32) std::copy_n(reinterpret_cast<const float *>(row.data()), input.size(), dequantized.data());
            else weights->to_float(row.data(), dequantized.data(), input.size());
            double expected = 0;
            for (size_t column = 0; column < input.size(); ++column) expected += double(dequantized[column]) * double(input[column]);
            float actual = 0, ordinary_cpu = 0;
            ggml_backend_tensor_get(value, &actual, index * sizeof(float), sizeof(float));
            cpu->vec_dot(int(input.size()), &ordinary_cpu, 0, row.data(), 0, packed.data(), 0, 1);
            if (!std::isfinite(expected) || !std::isfinite(actual) || !std::isfinite(ordinary_cpu)) return false;
            const auto backend_delta = double(actual) - expected, cpu_delta = double(ordinary_cpu) - expected;
            result.backend_max = std::max(result.backend_max, std::abs(backend_delta));
            result.cpu_max = std::max(result.cpu_max, std::abs(cpu_delta));
            result.backend_squared += backend_delta * backend_delta; result.cpu_squared += cpu_delta * cpu_delta;
            result.reference_squared += expected * expected; result.samples++;
        }
        *this = result; return true;
    }
};
