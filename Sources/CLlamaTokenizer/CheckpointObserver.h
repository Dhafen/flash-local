#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

// Fixed diagnostic vocabulary shared by the two isolated runtimes. No model
// tensor names or activation arrays are serialized by this observer.
namespace flash_local_checkpoint {
struct key { uint32_t stage = 0; int32_t layer = -1; };
inline key identify(std::string_view name) {
    if (name == "model.input_embed") return {1, -1};
    if (name == "result_norm") return {9, -1};
    const std::pair<std::string_view, uint32_t> prefixes[] = {
        {"attn_norm-", 2}, {"linear_attn_qkv_mixed-", 3}, {"conv_output_raw-", 4},
        {"linear_attn_out-", 5}, {"attn_output-", 5}, {"attn_post_norm-", 6},
        {"ffn_out-", 7}, {"l_out-", 8}};
    for (const auto & item : prefixes) {
        if (name.substr(0, item.first.size()) != item.first) continue;
        int32_t layer = -1;
        const auto parsed = std::from_chars(name.data() + item.first.size(), name.data() + name.size(), layer);
        if (parsed.ec != std::errc{} || parsed.ptr != name.data() + name.size() || layer < 0 || layer >= 40)
            return {};
        if ((item.second == 3 || item.second == 4) && layer != 0) return {};
        return {item.second, layer};
    }
    return {};
}
template <typename Callback>
inline bool emit(const ggml_tensor * tensor, Callback callback, void * user) {
    const auto id = identify(ggml_get_name(tensor));
    const auto count = ggml_nelements(tensor);
    if (!id.stage || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
        count < 1 || count > 16384) return false;
    std::vector<float> values(static_cast<size_t>(count));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) return false;
    return callback(id.stage, id.layer, values.data(), int32_t(count), user);
}
}
