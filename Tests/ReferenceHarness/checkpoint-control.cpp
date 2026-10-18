#include "FlashLlamaReference.h"
#include "../../Sources/CLlamaTokenizer/CheckpointObserver.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

// Bounded generated fixtures only; exercise the same C wrapper as Swift before
// building three Apple frameworks. No weights or activations are printed.
struct trace {
    std::map<uint64_t, std::vector<float>> points;
    size_t bytes = 0;
    std::string failure;
};
static bool capture(uint32_t stage, int32_t layer, const float * values, int32_t count, void * user) {
    auto & t = *static_cast<trace *>(user);
    if (!values || count < 1 || count > 16384 || t.bytes + size_t(count) * 4 > (4u << 20)) return false;
    const uint64_t key = uint64_t(stage) << 32 | uint32_t(layer);
    if (t.points.count(key)) {
        t.failure = "duplicate stage=" + std::to_string(stage) + " layer=" + std::to_string(layer);
        return false;
    }
    t.points[key] = std::vector<float>(values, values + count);
    t.bytes += size_t(count) * 4;
    return true;
}
static bool never_abort(void *) { return false; }
int main(int argc, char ** argv) {
    if (argc != 2 || std::filesystem::file_size(argv[1]) > (32u << 20)) return 2;
    using flash_local_checkpoint::identify;
    if (identify("attn_output-0").stage != 10 || identify("linear_attn_out-0").stage != 5 ||
        identify("attn_output-3").stage != 5 || identify("attn_output-1").stage != 0 ||
        identify("linear_attn_out-0 (reshaped)").stage != 0) return 10;
    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) return 3;
    const int tokens[] = {1, 2, 3, 15, 4};
    std::vector<std::vector<float>> expected;
    flr_context * plain = flr_create(fd, 32, never_abort, nullptr);
    if (!plain) { std::cerr << flr_last_error() << '\n'; close(fd); return 4; }
    const int size = flr_vocabulary_size(plain);
    if (size < 1 || size > 4096) { flr_free(plain); close(fd); return 5; }
    for (int token : tokens) {
        std::vector<float> row(size);
        if (flr_step(plain, token, row.data(), size) != size) {
            std::cerr << flr_last_error() << '\n'; flr_free(plain); close(fd); return 6;
        }
        expected.push_back(std::move(row));
    }
    flr_free(plain);
    trace t;
    flr_context * observed = flr_create_traced(fd, 32, never_abort, nullptr, capture, &t);
    close(fd);
    if (!observed) { std::cerr << flr_last_error() << '\n'; return 7; }
    size_t points = 0;
    for (size_t step = 0; step < expected.size(); ++step) {
        t.points.clear(); t.bytes = 0;
        std::vector<float> row(size);
        if (flr_step(observed, tokens[step], row.data(), size) != size) {
            std::cerr << flr_last_error() << " " << t.failure << " step=" << step << '\n'; flr_free(observed); return 8;
        }
        if (row != expected[step] || t.points.size() <= 20 || !t.points.count(uint64_t(10) << 32) ||
            !t.points.count(uint64_t(5) << 32) || !t.points.count((uint64_t(5) << 32) | 3)) {
            std::cerr << "Checkpoint replay differs or is incomplete\n"; flr_free(observed); return 9;
        }
        points += t.points.size();
    }
    flr_free(observed);
    std::cout << "ORNITH_CHECKPOINT_REFERENCE wrapperTrace=true exactLogits=true steps=5 points=" << points
              << " rawDeltaNetDistinct=true syntheticOnly=true physicalPhone=false\n";
}
