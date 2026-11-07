#include "ExpertReadWave.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
struct Request {
    int expert;
    int slot;
};

void require(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(message);
}

void check_reads() {
    constexpr size_t experts = 16, slots = 12, bytes = 64, prefix = 19;
    const std::vector<Request> demand = {{9, 1}, {2, 4}, {1, 9}, {0, 7}, {5, 0}, {4, 8}};
    auto ordered = demand;
    const auto ranges = flash_expert_read_ranges(ordered);
    require(ranges.size() == 3, "three exact file ranges expected");
    require(demand.front().expert == 9, "graph request order changed");

    std::array<unsigned char, prefix + experts * bytes> source{};
    std::array<unsigned char, slots * bytes> target{};
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t i = 0; i < bytes; ++i) source[prefix + expert * bytes + i] = expert * 11 + i;
    }
    target.fill(0xfd);
    char path[] = "/tmp/flash-expert-io-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "could not create fixture");
    unlink(path);
    require(write(fd, source.data(), source.size()) == static_cast<ssize_t>(source.size()), "fixture write failed");
    size_t read_bytes = 0;
    for (const auto range : ranges) {
        flash_read_expert_family(ordered.data() + range.begin, range.end - range.begin,
            prefix, bytes, target.data(), [&](size_t offset, iovec * iov, int count) {
                const auto size = preadv(fd, iov, count, static_cast<off_t>(offset));
                require(size == static_cast<ssize_t>(bytes * count), "short fixture read");
                read_bytes += size;
            });
    }
    close(fd);
    require(read_bytes == demand.size() * bytes, "demand-only byte ledger differs");
    for (const auto r : demand) {
        require(std::memcmp(target.data() + r.slot * bytes,
                            source.data() + prefix + r.expert * bytes, bytes) == 0,
                "wrong expert installed in GPU-visible slot");
    }
    require(target[2 * bytes] == 0xfd, "an unrequested slot changed");

    ordered.clear();
    for (int i = 11; i >= 0; --i) ordered.push_back({i, i});
    const auto capped = flash_expert_read_ranges(ordered);
    require(capped.size() == 2 && capped[0].end == 8 && capped[1].end == 12,
            "vector width must stay within the runtime reader limit");
}

void check_workers() {
    FlashExpertReadPool pool;
    for (int wave = 0; wave < 200; ++wave) {
        std::array<std::atomic<int>, 31> counts{};
        for (auto & count : counts) count.store(0);
        const int jobs = 1 + wave % 31;
        pool.run(jobs, 1 + wave % 4, [&](int i) { ++counts.at(i); });
        for (int i = 0; i < 31; ++i) require(counts[i] == (i < jobs ? 1 : 0), "wave lost or duplicated a read");
    }
    std::atomic<int> count{0};
    bool failed = false;
    try {
        pool.run(12, 4, [&](int i) {
            ++count;
            if (i == 3) throw std::runtime_error("failed read");
        });
    } catch (const std::runtime_error &) {
        failed = true;
    }
    require(failed && count == 12, "read errors must wait for other writes before propagating");
    pool.run(12, 4, [&](int) { ++count; });
    require(count == 24, "pool did not recover for a new wave");
}

void check_windows() {
    std::array<unsigned char, 8> bank{};
    require(flash_disjoint_expert_buffers(bank.data(), 4, bank.data() + 4, 4), "adjacent views should qualify");
    require(!flash_disjoint_expert_buffers(bank.data() + 3, 4, bank.data(), 4), "overlapping views must refuse retention");
    FlashExpertWindow saved;
    std::vector<int32_t> slots(4, -1), experts(16, -1);
    std::vector<uint64_t> ages(4, 0);
    uint64_t age = 0;
    saved.select(true, 8, slots, experts, ages, age);
    require(slots.size() == 8 && slots[0] == -1, "first demand must start cold");
    slots[7] = 12; experts[12] = 7; ages[7] = ++age;
    saved.select(false, 8, slots, experts, ages, age);
    require(slots.size() == 4 && experts[12] == -1, "prefill may only see staging slots");
    slots[0] = 4; experts[4] = 0; ages[0] = ++age;
    saved.select(true, 8, slots, experts, ages, age);
    require(slots[7] == 12 && experts[12] == 7 && ages[7] == 1 && age == 1 && experts[4] == -1,
            "prefill cleared or contaminated decode residency");
}

void benchmark() {
    constexpr size_t experts = 256, bytes = 16384, families = 3;
    std::vector<unsigned char> source(experts * bytes * families);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = (i / bytes * 173 + i / 256) & 255;
    }
    char path[] = "/tmp/flash-expert-bench-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "could not create benchmark fixture");
    unlink(path);
    require(write(fd, source.data(), source.size()) == static_cast<ssize_t>(source.size()), "fixture write failed");

    for (const int width : {8, 32}) {
        std::vector<unsigned char> legacy_output(bytes * width * families), candidate_output(legacy_output.size());
        FlashExpertReadPool pool;
        const auto run = [&](bool candidate, std::vector<unsigned char> & target) {
            uint64_t reads = 0;
            const auto started = std::chrono::steady_clock::now();
            for (int wave = 0; wave < 400; ++wave) {
                std::vector<Request> requests;
                for (int i = 0; i < width; ++i) {
                    const int expert = (wave * 11 + (i / 2) * 13 + i % 2) % int(experts);
                    requests.push_back({expert, width - 1 - i});
                }
                std::vector<FlashExpertReadRange> ranges;
                if (candidate) {
                    ranges = flash_expert_read_ranges(requests);
                } else {
                    for (int i = 0; i < width; ++i) ranges.push_back({size_t(i), size_t(i + 1)});
                }
                reads += ranges.size() * families;
                const auto read_range = [&](int i) {
                    const auto range = ranges[i];
                    for (size_t f = 0; f < families; ++f) {
                        flash_read_expert_family(requests.data() + range.begin, range.end - range.begin,
                            f * experts * bytes, bytes, target.data() + f * width * bytes,
                            [&](size_t offset, iovec * spans, int count) {
                                require(preadv(fd, spans, count, static_cast<off_t>(offset)) ==
                                        static_cast<ssize_t>(count * bytes), "short benchmark read");
                            });
                    }
                };
                if (candidate) {
                    pool.run(int(ranges.size()), 4, read_range);
                } else {
                    std::atomic<int> next{0};
                    std::array<std::thread, 4> threads;
                    std::atomic<bool> failed{false};
                    for (auto & thread : threads) {
                        thread = std::thread([&] {
                            for (int i = next++; i < width; i = next++) {
                                try { read_range(i); } catch (...) { failed = true; }
                            }
                        });
                    }
                    for (auto & thread : threads) thread.join();
                    require(!failed, "legacy benchmark read failed");
                }
            }
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started);
            std::cout << "width=" << width << " mode=" << (candidate ? "coalesced" : "legacy")
                      << " waveCount=400 logicalReads=" << reads << " milliseconds=" << elapsed.count() << '\n';
        };
        for (bool candidate : {false, true, true, false}) {
            run(candidate, candidate ? candidate_output : legacy_output);
        }
        require(legacy_output == candidate_output, "benchmark cache data differed");
    }
    close(fd);
}
}

int main(int argc, char ** argv) {
    try {
        check_reads();
        check_workers();
        check_windows();
        std::cout << "Ornith demand reads, exact scatter, and reusable worker barriers passed\n";
        if (argc == 2 && std::string(argv[1]) == "--benchmark") benchmark();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
