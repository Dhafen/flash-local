// Bounded real-weight slice check. No full model load or graph execution.
#include "ggml.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>
int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("q4_1-slice rows columns");
        const int rows = std::stoi(argv[2]), columns = std::stoi(argv[3]);
        if (rows < 1 || rows > 16 || columns < 32 || columns > 8192 || columns % 32)
            throw std::runtime_error("Slice dimensions exceed bounded control");
        const size_t stride = ggml_row_size(GGML_TYPE_Q4_1, columns), size = stride * rows;
        if (std::filesystem::file_size(argv[1]) != size || size > (2u << 20)) throw std::runtime_error("Wrong bounded slice size");
        std::ifstream in(argv[1], std::ios::binary);
        std::vector<uint8_t> packed(size); in.read(reinterpret_cast<char *>(packed.data()), size);
        if (!in) throw std::runtime_error("Cannot read weight slice");
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({16u << 20, nullptr, true}), ggml_free);
        ggml_cpu_init();
        const auto * type = ggml_get_type_traits(GGML_TYPE_Q4_1);
        const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_1);
        const auto * activation = ggml_get_type_traits_cpu(cpu->vec_dot_type);
        std::vector<float> x(columns), weights(size_t(rows) * columns), dots(rows);
        for (int i = 0; i < columns; ++i) x[i] = (i % 32 < 16 ? 1 : -1) * (127 - 7 * (i % 16));
        std::vector<uint8_t> quantized(ggml_row_size(cpu->vec_dot_type, columns));
        activation->from_float(x.data(), quantized.data(), columns);
        for (int row = 0; row < rows; ++row) {
            type->to_float(packed.data() + row * stride, weights.data() + row * columns, columns);
            cpu->vec_dot(columns, &dots[row], 0, packed.data() + row * stride, 0, quantized.data(), 0, 1);
        }
        for (float value : weights) if (!std::isfinite(value)) throw std::runtime_error("Non-finite decoded real weights");
        for (float value : dots) if (!std::isfinite(value)) throw std::runtime_error("Non-finite real-weight projection");
        std::cout << nlohmann::json{{"rows", rows}, {"columns", columns}, {"weights", weights}, {"dots", dots},
            {"activation", "integer-zero-sum-blocks-exact-q8_1"}, {"trainedGraphExecuted", false}}.dump() << '\n';
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
