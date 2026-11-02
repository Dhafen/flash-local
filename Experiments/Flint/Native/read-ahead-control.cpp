// Standalone bounded advisory worker test; intentionally never loads a model.
#include "FlintExpertReadAhead.h"
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <unistd.h>

int main() {
    char name[] = "/tmp/flint-advice-XXXXXX";
    const int fd = mkstemp(name);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    unlink(name);
    if (ftruncate(fd, 1 << 20) != 0) {
        close(fd);
        throw std::runtime_error("ftruncate failed");
    }
    {
        std::vector<FlintExpertReadAhead::Family> families{
            {2, 0, 4096, 8}, {2, 32768, 4096, 8}, {2, 65536, 4096, 8}
        };
        FlintExpertReadAhead worker(fd, families);
        const int32_t routed[] = {0, 1, 2, 3, 4, 5};
        const int32_t invalid[] = {-1, 99};
        worker.suggest(2, routed, 6);
        worker.suggest(2, invalid, 2);
        worker.suggest(100, routed, 6);
        // Malformed family metadata must not wrap the byte offset and issue
        // read advice against an unrelated part of the model file.
        const uint64_t max_signed_offset = uint64_t(INT64_MAX);
        FlintExpertReadAhead overflow_worker(fd, {
            {7, max_signed_offset - 100, 64, 8},
            {8, 0, UINT64_MAX, 8}
        });
        const int32_t overflow_expert[] = {2};
        overflow_worker.suggest(7, overflow_expert, 1);
        overflow_worker.suggest(8, overflow_expert, 1);
        const auto overflow_stats = overflow_worker.snapshot();
        if (overflow_stats.requests != 2 || overflow_stats.drops != 2 ||
            overflow_stats.accepted != 0)
            throw std::runtime_error("Overflowing expert offsets were not rejected");
        overflow_worker.stop();
        // This flood must remain bounded; stop() clears pending requests and
        // joins the only worker before the file descriptor is closed.
        for (int n = 0; n < 100; ++n) worker.suggest(2, routed, 6);
        // The earlier queue-only test could stop before the worker made any
        // advisory system call. Give it a bounded opportunity to execute,
        // then prove that at least one real attempt completed.
        for (int n = 0; n < 100; ++n) {
            const auto observed = worker.snapshot();
            if (observed.advised_bytes > 0 || observed.failures > 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        worker.stop();
        auto stats = worker.snapshot();
        if (stats.advised_bytes == 0 && stats.failures == 0)
            throw std::runtime_error("Read-ahead worker never attempted an advisory operation");
        if (stats.requests != 6 + 2 + 6 + 600 || stats.accepted == 0 || stats.drops < 8)
            throw std::runtime_error("Read-ahead queue rejected its bounded test invariants");
        std::printf("FLINT_ADVICE_CONTROL accepted=%llu drops=%llu duplicates=%llu advised=%llu failures=%llu joined=true\n",
            (unsigned long long)stats.accepted, (unsigned long long)stats.drops,
            (unsigned long long)stats.duplicates, (unsigned long long)stats.advised_bytes,
            (unsigned long long)stats.failures);
    }
    close(fd);
    return 0;
}
