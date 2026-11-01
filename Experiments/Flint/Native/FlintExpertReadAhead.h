#pragma once
// Bounded, advisory expert read-ahead. This helper NEVER returns expert data or
// modifies routing: the pinned llama.cpp slot bank remains authoritative.
// Apple F_RDADVISE asks the kernel to warm the file cache; it is not proof of
// physical NAND I/O. The caller owns the verified descriptor until join().
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <vector>
#include <sys/types.h>

class FlintExpertReadAhead {
public:
    struct Family {
        int layer;
        uint64_t offset;
        uint64_t bytes_per_expert;
        int experts;
    };
    struct Stats {
        uint64_t requests = 0, accepted = 0, duplicates = 0;
        uint64_t drops = 0, advised_bytes = 0, failures = 0;
    };
    static constexpr size_t max_pending = 48;
    FlintExpertReadAhead(int descriptor, std::vector<Family> families)
        : fd_(descriptor), families_(std::move(families)), worker_([this] { run(); }) {}
    FlintExpertReadAhead(const FlintExpertReadAhead &) = delete;
    FlintExpertReadAhead & operator=(const FlintExpertReadAhead &) = delete;
    ~FlintExpertReadAhead() { stop(); }

    // Nonblocking: this is an *advisory* hint, never part of model correctness.
    void suggest(int layer, const int32_t * ids, size_t count) {
        if (!ids || count == 0 || count > 6) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) return;
        for (size_t k = 0; k < count; ++k) {
            const int expert = ids[k];
            ++stats_.requests;
            bool valid = false;
            for (const auto & f : families_) {
                if (f.layer == layer && expert >= 0 && expert < f.experts) {
                    valid = true;
                    const Job job{f.offset + uint64_t(expert) * f.bytes_per_expert, f.bytes_per_expert};
                    if (job.bytes == 0 || job.offset > uint64_t(INT64_MAX) ||
                        job.bytes > uint64_t(INT64_MAX) - job.offset) { ++stats_.drops; continue; }
                    bool duplicate = false;
                    for (const auto & queued : queue_) {
                        if (queued.offset == job.offset && queued.bytes == job.bytes) { duplicate = true; break; }
                    }
                    if (duplicate) { ++stats_.duplicates; continue; }
                    if (queue_.size() >= max_pending) { ++stats_.drops; continue; }
                    queue_.push_back(job);
                    ++stats_.accepted;
                }
            }
            if (!valid) ++stats_.drops;
        }
        cv_.notify_one();
    }

    Stats snapshot() const { std::lock_guard<std::mutex> lock(mu_); return stats_; }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!stopping_) { stopping_ = true; queue_.clear(); }
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    struct Job { uint64_t offset; uint64_t bytes; };
    const int fd_;
    const std::vector<Family> families_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    bool stopping_ = false;
    Stats stats_;
    std::thread worker_;
    void run() {
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_) return;
                job = queue_.front();
                queue_.pop_front();
            }
            bool ok = false;
#if defined(__APPLE__) && defined(F_RDADVISE)
            // Apple radvisory is bounded by the verified tensor region and the
            // documented rad_count field; do not allocate or read model bytes.
            radvisory advice{};
            advice.ra_offset = static_cast<off_t>(job.offset);
            advice.ra_count = static_cast<int>(std::min<uint64_t>(job.bytes, INT32_MAX));
            ok = fcntl(fd_, F_RDADVISE, &advice) == 0;
#elif defined(POSIX_FADV_WILLNEED)
            ok = posix_fadvise(fd_, static_cast<off_t>(job.offset),
                               static_cast<off_t>(job.bytes), POSIX_FADV_WILLNEED) == 0;
#endif
            std::lock_guard<std::mutex> lock(mu_);
            if (ok) stats_.advised_bytes += job.bytes;
            else ++stats_.failures;
        }
    }
};
