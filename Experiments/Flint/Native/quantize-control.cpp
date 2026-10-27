// Canonical GGML Q4_1 encoding of SMALL random fixtures only. No model conversion.
#include "ggml.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <regex>
#include <stdexcept>
#include <cstring>

int main(int argc, char ** argv) {
    try {
        namespace fs = std::filesystem;
        if (argc != 3 || fs::file_size(argv[1]) > (64u << 20)) throw std::runtime_error("small-f32.gguf output-directory (64 MiB input cap)");
        fs::path directory = argv[2]; fs::create_directories(directory);
        ggml_context * tensors = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> input(gguf_init_from_file(argv[1], {false, &tensors}), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(tensors, ggml_free);
        if (!input || !tensors) throw std::runtime_error("Fixture GGUF load failed");
        const int64_t count = gguf_get_n_tensors(input.get());
        std::unique_ptr<ggml_context, decltype(&ggml_free)> dst(ggml_init({size_t(count + 8) * ggml_tensor_overhead(), nullptr, true}), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
        gguf_set_kv(output.get(), input.get());
        std::vector<std::vector<uint8_t>> data;
        for (int64_t i = 0; i < count; ++i) {
            const std::string name = gguf_get_tensor_name(input.get(), i);
            auto * tensor = ggml_get_tensor(tensors, name.c_str());
            const bool quantize = ggml_n_dims(tensor) >= 2 && name != "token_embd.weight" && name.find("ffn_gate_inp") == std::string::npos;
            auto * target = ggml_new_tensor(dst.get(), quantize ? GGML_TYPE_Q4_1 : GGML_TYPE_F32, ggml_n_dims(tensor), tensor->ne);
            ggml_set_name(target, name.c_str());
            data.emplace_back(ggml_nbytes(target));
            if (quantize) {
                const size_t n = ggml_quantize_chunk(GGML_TYPE_Q4_1, static_cast<const float *>(tensor->data), data.back().data(),
                                                    0, ggml_nrows(tensor), tensor->ne[0], nullptr);
                if (n != data.back().size()) throw std::runtime_error("Quantized byte size mismatch");
            } else std::memcpy(data.back().data(), tensor->data, data.back().size());
            gguf_add_tensor(output.get(), target);
        }
        const auto path = directory / "synthetic-flint-q4_1.gguf";
        if (!gguf_write_to_file(output.get(), path.c_str(), true)) throw std::runtime_error("Cannot write control GGUF");
        std::fstream stream(path, std::ios::in | std::ios::out | std::ios::binary);
        auto entries = nlohmann::json::array();
        const auto expert_key = gguf_find_key(output.get(), "flint.expert_count");
        const auto experts = gguf_get_val_u32(output.get(), expert_key);
        std::regex routed("^blk\\.([0-9]+)\\.(ffn_(gate|up|down)_exps)\\.weight$");
        for (int64_t i = 0; i < count; ++i) {
            const size_t offset = gguf_get_meta_size(output.get()) + gguf_get_tensor_offset(output.get(), i);
            stream.seekp(offset); stream.write(reinterpret_cast<const char *>(data[i].data()), data[i].size());
            const std::string name = gguf_get_tensor_name(output.get(), i);
            std::smatch match;
            if (std::regex_match(name, match, routed)) entries.push_back({{"layer", std::stoi(match[1])}, {"tensor_name", name},
                {"tensor_family", match[2].str()}, {"quant_type", "q4_1"}, {"expert_count", experts},
                {"exact_byte_length", data[i].size()}, {"bytes_per_expert", data[i].size() / experts},
                {"expert_stride", data[i].size() / experts}, {"repacked_offset", offset}, {"repacked_file", path.filename().string()}});
        }
        stream.close(); if (!stream) throw std::runtime_error("Control payload write failed");
        std::ofstream manifest(directory / "manifest.json");
        manifest << nlohmann::json{{"schema_version", 1}, {"experts_kind", "omni_moe_gguf"}, {"layout", "gguf_tensor_major"},
            {"model", {{"expert_count", experts}, {"expert_used_count", 6}}}, {"entries", entries}}.dump(2) << std::endl;
        if (!manifest) throw std::runtime_error("Control manifest write failed");
        std::cout << "Canonical GGML Q4_1 synthetic fixture ready; no trained weights" << std::endl;
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << std::endl; return 1; }
}
