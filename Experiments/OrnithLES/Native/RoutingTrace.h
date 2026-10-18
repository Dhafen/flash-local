#pragma once
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <vector>

// Observer only. Never writes tensor data or participates in expert selection.
// The sequential digest preserves the original single-token report format.
// The canonical sum hashes (absolute position, layer, rank, expert), allowing
// token-major decode and layer-major prefill to be compared without reordering
// any actual computation. It is an aggregate checksum, not an attestation.
struct flash_local_routing_trace {
    uint64_t sequential = 14695981039346656037ULL;
    uint64_t canonical = 0, decisions = 0;

    template <typename ReadRow>
    bool append(std::string_view name, int position, int64_t ranks, int64_t rows,
                size_t row_stride, uint32_t experts, ReadRow read) {
        constexpr std::string_view prefix = "ffn_moe_topk-";
        if (name.substr(0, prefix.size()) != prefix || position < 0 ||
            ranks < 1 || ranks > experts || experts > 256 || rows < 1 || rows > 32 ||
            uint64_t(position) + uint64_t(rows) > 65536 ||
            row_stride < size_t(ranks) * sizeof(int32_t) ||
            (rows > 1 && row_stride > (std::numeric_limits<size_t>::max() - size_t(ranks) * sizeof(int32_t)) /
                size_t(rows - 1))) return false;
        uint32_t layer = 0;
        const auto parsed = std::from_chars(name.data() + prefix.size(), name.data() + name.size(), layer);
        if (parsed.ec != std::errc{} || parsed.ptr != name.data() + name.size() || layer >= 256) return false;

        std::vector<int32_t> ids(static_cast<size_t>(ranks));
        uint64_t ordered = sequential, normalized = canonical;
        for (int64_t row = 0; row < rows; ++row) {
            // ggml_argsort_top_k is a strided view of the full expert sort.
            read(ids.data(), size_t(row) * row_stride, ids.size() * sizeof(int32_t));
            for (uint32_t rank = 0; rank < ids.size(); ++rank) {
                const int32_t expert = ids[rank];
                if (expert < 0 || uint32_t(expert) >= experts) return false;
                ordered ^= uint32_t(expert); ordered *= 1099511628211ULL;
                uint64_t tuple = 14695981039346656037ULL;
                for (uint32_t value : {uint32_t(position + row), layer, rank, uint32_t(expert)}) {
                    for (unsigned shift = 0; shift < 32; shift += 8) {
                        tuple ^= (value >> shift) & 255U; tuple *= 1099511628211ULL;
                    }
                }
                normalized += tuple;
            }
        }
        sequential = ordered; canonical = normalized; decisions += uint64_t(rows * ranks);
        return true;
    }
};
