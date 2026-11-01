// Standalone bounded advisory worker test; intentionally never loads a model.
#include "FlintExpertReadAhead.h"
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
        // This flood must remain bounded; stop() clears pending requests and
        // joins the only worker before the file descriptor is closed.
        for (int n = 0; n < 100; ++n) worker.suggest(2, routed, 6);
        worker.stop();
        auto stats = worker.snapshot();
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
