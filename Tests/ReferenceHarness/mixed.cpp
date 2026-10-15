#include "ggml.h"
#include "gguf.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// Test fixture converter only. Uses the pinned canonical quantizers, rather
// than an independently invented encoder or any trained checkpoint.
int main(int argc, char ** argv) {
    if (argc != 3 || std::filesystem::file_size(argv[1]) > (32u << 20)) return 2;
    ggml_context * data = nullptr;
    std::cerr << "MIXED_FIXTURE loading bounded random F32 source\n";
    gguf_context * file = gguf_init_from_file(argv[1], {false, &data});
    if (!file || !data) return 3;
    const auto key = gguf_find_key(file, "general.name");
    if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(file, key)) != "synthetic-graph-test" ||
        gguf_get_n_tensors(file) != 113) return 4;
    // Reader tensor metadata is not a fully initialized ggml_tensor. Build a
    // fresh writer from the real allocated tensors, including valid buffer
    // pointers, instead of reusing the reader's internal tensor structs.
    gguf_context * encoded = gguf_init_empty();
    gguf_set_kv(encoded, file);
    std::vector<std::vector<uint8_t>> packed(gguf_get_n_tensors(file));
    std::vector<bool> seen(GGML_TYPE_COUNT, false);
    for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
        const std::string name = gguf_get_tensor_name(file, i);
        ggml_tensor * tensor = ggml_get_tensor(data, name.c_str());
        if (!tensor || tensor->type != GGML_TYPE_F32) return 5;
        gguf_add_tensor(encoded, tensor);
        ggml_type type = GGML_TYPE_F32;
        if (name.find("ffn_gate_exps.weight") != std::string::npos) type = GGML_TYPE_Q4_K;
        else if (name.find("ffn_up_exps.weight") != std::string::npos) type = GGML_TYPE_Q5_0;
        else if (name.find("ffn_down_exps.weight") != std::string::npos || name == "output.weight") type = GGML_TYPE_Q6_K;
        else if (name == "per_layer_token_embd.weight") type = GGML_TYPE_Q4_0;
        else if (name == "token_embd.weight" || name.find("ffn_gate_shexp.weight") != std::string::npos) type = GGML_TYPE_Q8_0;
        else if (name.find("hc_down.weight") != std::string::npos || name.find("hc_attn_down.weight") != std::string::npos ||
                 name.find("hc_ffn_down.weight") != std::string::npos) type = GGML_TYPE_F16;
        else if (name.find("ffn_gate_inp.weight") != std::string::npos) type = GGML_TYPE_BF16;
        seen[type] = true;
        if (type == GGML_TYPE_F32) {
            // The reader binds payloads on the separate ggml tensors; the GGUF
            // writer needs explicit pointers even for unchanged F32 tensors.
            gguf_set_tensor_data(encoded, name.c_str(), tensor->data);
            continue;
        }
        if (tensor->ne[0] % ggml_blck_size(type)) return 6;
        const int64_t rows = ggml_nelements(tensor) / tensor->ne[0];
        const size_t bytes = ggml_row_size(type, tensor->ne[0]) * rows;
        std::cerr << "MIXED_FIXTURE tensor=" << name << " type=" << ggml_type_name(type)
                  << " columns=" << tensor->ne[0] << " rows=" << rows << " bytes=" << bytes << '\n';
        packed[i].resize(bytes);
        if (ggml_quantize_chunk(type, static_cast<const float *>(tensor->data), packed[i].data(),
                                0, rows, tensor->ne[0], nullptr) != bytes) return 7;
        gguf_set_tensor_type(encoded, name.c_str(), type);
        gguf_set_tensor_data(encoded, name.c_str(), packed[i].data());
    }
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0,
                      GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        if (!seen[type]) return 8;
    }
    std::cerr << "MIXED_FIXTURE writing encoded tensors\n";
    const bool written = gguf_write_to_file(encoded, argv[2], false);
    gguf_free(encoded); gguf_free(file); ggml_free(data);
    std::cout << "REAM_MIXED_FIXTURE canonicalQuantizers=true types=8 trainedWeights=false\n";
    return written ? 0 : 9;
}
