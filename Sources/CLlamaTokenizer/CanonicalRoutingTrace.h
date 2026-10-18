#pragma once
#include <charconv>
#include <array>
#include <cstddef>
#include <cstdint>
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
    uint64_t canonical = 0, decisions = 0, expert_sets = 0;

    static uint64_t mix(uint64_t tuple) {
        tuple += 0x9e3779b97f4a7c15ULL;
        tuple = (tuple ^ (tuple >> 30)) * 0xbf58476d1ce4e5b9ULL;
        tuple = (tuple ^ (tuple >> 27)) * 0x94d049bb133111ebULL;
        return tuple ^ (tuple >> 31);
    }

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
        uint64_t ordered = sequential, normalized = canonical, sets = expert_sets;
        for (int64_t row = 0; row < rows; ++row) {
            // ggml_argsort_top_k is a strided view of the full expert sort.
            read(ids.data(), size_t(row) * row_stride, ids.size() * sizeof(int32_t));
            std::array<bool, 256> seen{};
            for (uint32_t rank = 0; rank < ids.size(); ++rank) {
                const int32_t expert = ids[rank];
                if (expert < 0 || uint32_t(expert) >= experts || seen[size_t(expert)]) return false;
                seen[size_t(expert)] = true;
                ordered ^= uint32_t(expert); ordered *= 1099511628211ULL;
                // Each field has a disjoint, bounded bit range. Avalanche the
                // complete tuple before summation: summing raw FNV tuples can
                // cancel correlated expert changes in adjacent layers.
                uint64_t tuple = uint64_t(position + row) | (uint64_t(layer) << 16) |
                    (uint64_t(rank) << 24) | (uint64_t(expert) << 32);
                normalized += mix(tuple);
                // A second checksum ignores only rank, retaining the absolute
                // token position, layer and every distinct selected expert.
                // No IDs or model arithmetic are sorted/changed by this observer.
                sets += mix(uint64_t(position + row) | (uint64_t(layer) << 16) |
                    (uint64_t(expert) << 32));
            }
        }
        sequential = ordered; canonical = normalized; expert_sets = sets; decisions += uint64_t(rows * ranks);
        return true;
    }
};
