#include "../../Experiments/OrnithLES/Native/ProjectionProbe.h"
#include "ggml-alloc.h"
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    try {
        ggml_cpu_init();
        ggml_init_params params{64 << 10, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        if (!ctx) throw std::runtime_error("Probe context allocation");
        constexpr int columns = 256, rows = 16;
        auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, columns, rows);
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, columns, 1);
        auto * y = ggml_mul_mat(ctx.get(), w, x);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
            ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()), ggml_backend_buffer_free);
        if (!storage) throw std::runtime_error("Probe CPU allocation");
        std::vector<float> source(columns * rows), input(columns), output(rows), decoded(columns);
        for (size_t i = 0; i < source.size(); ++i) source[i] = std::cos(float(i) * .091f) * .2f + .004f * float(i % 7);
        for (int i = 0; i < columns; ++i) input[i] = std::sin(float(i) * .117f) * .7f + .03f * std::cos(float(i) * 1.7f);
        std::vector<uint64_t> packed((ggml_nbytes(w) + 7) / 8);
        if (ggml_quantize_chunk(GGML_TYPE_Q4_K, source.data(), packed.data(), 0, rows, columns, nullptr) != ggml_nbytes(w))
            throw std::runtime_error("Probe canonical quantizer size");
        ggml_backend_tensor_set(w, packed.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        const auto row_bytes = ggml_row_size(w->type, columns);
        for (int row = 0; row < rows; ++row) {
            ggml_get_type_traits(w->type)->to_float(reinterpret_cast<const char *>(packed.data()) + row * row_bytes, decoded.data(), columns);
            double sum = 0;
            for (int column = 0; column < columns; ++column) sum += double(decoded[column]) * double(input[column]);
            output[row] = float(sum);
        }
        ggml_backend_tensor_set(y, output.data(), 0, output.size() * sizeof(float));
        flash_local_projection_probe exact;
        if (!exact.run(y) || exact.samples != 16 || exact.backend_max > 1e-6 ||
            exact.cpu_max < 1e-6 || exact.cpu_max < 10 * exact.backend_max)
            throw std::runtime_error("Matched-input probe failed to separate CPU operand rounding");
        output[0] += 1;
        ggml_backend_tensor_set(y, output.data(), 0, output.size() * sizeof(float));
        flash_local_projection_probe corrupted;
        if (!corrupted.run(y) || corrupted.backend_max < .9)
            throw std::runtime_error("Probe did not detect a changed backend result");
        std::cout << "ORNITH_PROJECTION_PROBE matchedInput=true cpuRoundingDistinguished=true changedResultDetected=true syntheticOnly=true gpuExecuted=false"
                  << " exactMax=" << exact.backend_max << " cpuMax=" << exact.cpu_max << '\n';
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
