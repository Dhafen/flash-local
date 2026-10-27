#include "FlashFlint.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>

static void require(bool ok, const char * text) { if (!ok) throw std::runtime_error(text); }
int main(int argc, char ** argv) {
    flf_context * c = nullptr;
    try {
        if (argc != 3) throw std::runtime_error("small-flint.gguf empty-private-directory");
        require(std::filesystem::file_size(argv[1]) < (64u << 20), "Control admits small random fixtures only");
        std::filesystem::create_directories(argv[2]);
        const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
        require(fd >= 0, "Cannot open fixture");
        flf_options o{256, 32, 4, 2, 5500000000ULL, 400000000ULL, false, true};
        auto rejected = o; rejected.process_ceiling_bytes = 1024;
        require(flf_open(fd, argv[2], rejected, nullptr, nullptr, nullptr) == nullptr, "Over-budget load accepted");
        require(std::filesystem::is_empty(argv[2]), "Rejected load left a manifest");
        c = flf_open(fd, argv[2], o, nullptr, nullptr, nullptr); close(fd);
        if (!c) throw std::runtime_error(flf_last_error());
        // A second model must not replace/free the first model's ownership.
        require(flf_open(-1, argv[2], o, nullptr, nullptr, nullptr) == nullptr, "Duplicate load accepted");
        require(flf_vocab_size(c) == 32, "Wrong fixture vocabulary");
        std::vector<float> control(32), state_only(32);
        const int32_t prefix[] = {1, 2, 3, 4};
        require(flf_evaluate(c, prefix, 4, control.data(), 32) == 0, flf_last_error());
        const int32_t last[] = {5};
        require(flf_evaluate(c, last, 1, control.data(), 32) == 0, flf_last_error());
        require(flf_snapshot(c).cached_tokens == 5, "Append KV position was lost");
        require(flf_reset(c) == 0, "Reset failed");
        require(flf_evaluate(c, prefix, 4, nullptr, 0) == 0, flf_last_error());
        require(flf_evaluate(c, last, 1, state_only.data(), 32) == 0, flf_last_error());
        for (int i = 0; i < 32; ++i) require(std::isfinite(control[i]) && std::abs(control[i] - state_only[i]) < 1e-6f, "State-only prefill changed logits");
        require(flf_snapshot(c).state_only_tokens == 4 && flf_snapshot(c).cached_tokens == 5 && flf_snapshot(c).io_bytes > 0, "Missing real prefill/KV/I/O statistics");
        flf_cancel(c);
        require(flf_evaluate(c, last, 1, state_only.data(), 32) != 0, "Cancellation ignored");
        require(flf_reset(c) == 0 && flf_snapshot(c).cached_tokens == 0, "Cancellation recovery did not reset KV");
        require(flf_evaluate(c, prefix, 4, control.data(), 32) == 0, flf_last_error());
        flf_close(c); c = nullptr;
        require(std::filesystem::is_empty(argv[2]), "Private manifest not cleaned on unload");
        std::cout << "FLINT_API_CONTROL memoryAdmission=true duplicateLoadRejected=true descriptorRetained=true stateOnlyPrefill=true kvAppend=true reset=true cancellation=true unload=true\n";
        return 0;
    } catch (const std::exception & e) {
        if (c) flf_close(c);
        std::cerr << "Flint API control: " << e.what() << std::endl; return 1;
    }
}
