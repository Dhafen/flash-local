#pragma once
// Bounded, advisory expert read-ahead. This helper NEVER returns expert data or
// modifies routing: the pinned llama.cpp slot bank remains authoritative.
// Apple F_RDADVISE asks the kernel to warm the file cache; it is not proof of
// physical NAND I/O. The caller owns the verified descriptor until join().
#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <utility>
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
        uint64_t drops = 0, advised_bytes = 0, failures = 0, coalesced = 0;
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
                    // Validate the multiplication and addition BEFORE computing a file offset.
                    // A malformed manifest must never wrap into a different tensor.
                    const uint64_t limit = uint64_t(INT64_MAX);
                    if (f.bytes_per_expert == 0 || f.offset > limit ||
                        uint64_t(expert) > (limit - f.offset) / f.bytes_per_expert) {
                        ++stats_.drops; continue;
                    }
                    const Job job{f.offset + uint64_t(expert) * f.bytes_per_expert, f.bytes_per_expert};
                    if (job.bytes > limit - job.offset) { ++stats_.drops; continue; }
                    // The same expert may be requested again before a previous
                    // coalesced hint reaches the OS. Count it as a duplicate,
                    // not another disk advice request.
                    const auto covers = [&job](const Job & prior) {
                        return job.offset >= prior.offset && job.bytes <= prior.bytes &&
                            job.offset - prior.offset <= prior.bytes - job.bytes;
                    };
                    bool duplicate = in_flight_ && covers(active_job_);
                    for (const auto & queued : queue_) {
                        if (covers(queued)) { duplicate = true; break; }
                    }
                    if (duplicate) { ++stats_.duplicates; continue; }
                    // Consecutive routed expert extents can share one advisory
                    // hint. Merge only adjacent, pending extents; never change
                    // real expert reads, routing or the pinned model file.
                    if (!queue_.empty()) {
                        auto & tail = queue_.back();
                        if (tail.bytes <= uint64_t(INT32_MAX) &&
                            job.bytes <= uint64_t(INT32_MAX) - tail.bytes &&
                            tail.offset <= limit - tail.bytes &&
                            tail.offset + tail.bytes == job.offset) {
                            tail.bytes += job.bytes;
                            ++stats_.coalesced;
                            ++stats_.accepted;
                            continue;
                        }
                    }
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
    bool in_flight_ = false;
    Job active_job_{};
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
                active_job_ = job;
                in_flight_ = true;
            }
            bool ok = false;
            uint64_t requested_bytes = 0;
#if defined(__APPLE__) && defined(F_RDADVISE)
            // F_RDADVISE takes an int count: report the actual submitted
            // advisory range, not the possibly larger tensor stride.
            radvisory advice{};
            advice.ra_offset = static_cast<off_t>(job.offset);
            requested_bytes = std::min<uint64_t>(job.bytes, INT32_MAX);
            advice.ra_count = static_cast<int>(requested_bytes);
            ok = fcntl(fd_, F_RDADVISE, &advice) == 0;
#elif defined(POSIX_FADV_WILLNEED)
            requested_bytes = job.bytes;
            ok = posix_fadvise(fd_, static_cast<off_t>(job.offset),
                               static_cast<off_t>(job.bytes), POSIX_FADV_WILLNEED) == 0;
#endif
            std::lock_guard<std::mutex> lock(mu_);
            in_flight_ = false;
            if (ok) stats_.advised_bytes += requested_bytes;
            else ++stats_.failures;
        }
    }
};
