// Functional candidate controls, never an iPhone throughput benchmark.
#include "BoundedIOPool.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <numeric>
#include <unistd.h>

using flash_local::bounded_io_pool;
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        char name[] = "/tmp/flash-local-io-XXXXXX";
        const int descriptor = mkstemp(name);
        require(descriptor >= 0, "Cannot create synthetic source");
        unlink(name);
        struct close_file { int descriptor; ~close_file() { close(descriptor); } } owned{descriptor};
        constexpr std::size_t row_bytes = 4096, source_rows = 256 * 3;
        std::vector<unsigned char> source(row_bytes * source_rows);
        for (std::size_t index = 0; index < source.size(); ++index) source[index] = (index * 17 + index / row_bytes * 13) % 251;
        require(write(descriptor, source.data(), source.size()) == static_cast<ssize_t>(source.size()), "Short synthetic write");
        std::size_t total_jobs = 0, total_waves = 0;
        for (unsigned readers : {1u, 2u, 4u, 8u}) {
            {
                bounded_io_pool pool(readers);
                for (std::size_t round = 0; round < 64; ++round) {
                    // Three tensor families and deterministic changing expert
                    // IDs. Each job writes a distinct range of an existing buffer.
                    const std::size_t count = round % 4 == 0 ? 32 : 8;
                    std::vector<unsigned char> target(count * row_bytes * 3, 0);
                    std::vector<std::atomic<int>> seen(count);
                    for (auto & item : seen) item.store(0);
                    pool.run(count, [&](std::size_t index) {
                        seen[index].fetch_add(1);
                        const auto expert = (index * 29 + round * 11) % 256;
                        for (std::size_t family = 0; family < 3; ++family) {
                            auto * output = target.data() + (index * 3 + family) * row_bytes;
                            const auto offset = (family * 256 + expert) * row_bytes;
                            require(pread(descriptor, output, row_bytes, offset) == static_cast<ssize_t>(row_bytes), "Short pread");
                        }
                    });
                    for (std::size_t index = 0; index < count; ++index) {
                        require(seen[index].load() == 1, "Dropped or duplicated expert read");
                        const auto expert = (index * 29 + round * 11) % 256;
                        for (std::size_t family = 0; family < 3; ++family) {
                            const auto * actual = target.data() + (index * 3 + family) * row_bytes;
                            const auto * expected = source.data() + (family * 256 + expert) * row_bytes;
                            require(std::memcmp(actual, expected, row_bytes) == 0, "Expert byte mismatch");
                        }
                    }
                    total_jobs += count; ++total_waves;
                }
                require(pool.thread_launches() == readers - 1, "Threads recreated per wave");
                require(pool.completed_waves() == 64, "Missing wave completion");
                std::atomic<unsigned> attempted{0};
                bool failed = false;
                try { pool.run(32, [&](std::size_t index) { attempted.fetch_add(1); if (index == 7) throw std::runtime_error("injected read failure"); }); }
                catch (const std::runtime_error &) { failed = true; }
                require(failed && attempted.load() == 32, "Failure did not drain all readers before propagation");
                unsigned resumed = 0;
                pool.run(1, [&](std::size_t) { ++resumed; });
                require(resumed == 1, "A prior read failure poisoned reuse");
                pool.run(0, [](std::size_t) { throw std::runtime_error("empty wave ran"); });
                bool oversize = false;
                try { pool.run(257, [](std::size_t) {}); } catch (const std::invalid_argument &) { oversize = true; }
                require(oversize, "Unbounded wave admitted");
                bool overlap = false;
                try { pool.run(1, [&](std::size_t) { pool.run(1, [](std::size_t) {}); }); }
                catch (const std::logic_error &) { overlap = true; }
                require(overlap, "Nested wave was not rejected");
            }
            require(bounded_io_pool::live_helpers() == 0, "Helpers survived their owner");
        }
        for (unsigned readers : {0u, 9u}) {
            bool refused = false;
            try { bounded_io_pool invalid(readers); } catch (const std::invalid_argument &) { refused = true; }
            require(refused, "Unbounded worker count admitted");
        }
        std::printf("ORNITH_IO_POOL readers=1,2,4,8 waves=%zu expertJobs=%zu byteIdentical=true exactlyOnce=true threadsReused=true failuresDrained=true ownersReleased=true modelLoaded=false physicalPhone=false\n", total_waves, total_jobs);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "I/O candidate control failed: %s\n", error.what()); return 1;
    }
}
