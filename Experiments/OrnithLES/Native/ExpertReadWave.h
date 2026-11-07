#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <sys/uio.h>
#include <thread>
#include <utility>
#include <vector>

struct FlashExpertReadRange {
    size_t begin;
    size_t end;
};

constexpr size_t flash_expert_read_width = 8;

inline bool flash_disjoint_expert_buffers(const void * a, size_t a_bytes, const void * b, size_t b_bytes) {
    if (!a || !b || !a_bytes || !b_bytes) return false;
    const auto left = reinterpret_cast<uintptr_t>(a);
    const auto right = reinterpret_cast<uintptr_t>(b);
    return left <= right ? right - left >= a_bytes : left - right >= b_bytes;
}

// The inactive slot map belongs to a distinct tensor region. Switching graphs
// swaps only small maps and their LRU clock. No expert payload moves.
class FlashExpertWindow {
public:
    void select(bool decode, int32_t width, std::vector<int32_t> & slots,
                std::vector<int32_t> & experts, std::vector<uint64_t> & ages, uint64_t & age) {
        if (decode == decode_) return;
        if (slots_.empty()) {
            slots_.assign(width, -1);
            experts_.assign(experts.size(), -1);
            ages_.assign(width, 0);
        }
        slots.swap(slots_);
        experts.swap(experts_);
        ages.swap(ages_);
        std::swap(age, age_);
        decode_ = decode;
    }

private:
    std::vector<int32_t> slots_, experts_;
    std::vector<uint64_t> ages_;
    uint64_t age_ = 0;
    bool decode_ = false;
};

// Sort an I/O-only copy. The graph's expert order and slot mapping are authoritative.
template <class Request>
std::vector<FlashExpertReadRange> flash_expert_read_ranges(std::vector<Request> & ordered) {
    std::sort(ordered.begin(), ordered.end(), [](const Request & a, const Request & b) {
        return a.expert < b.expert;
    });
    std::vector<FlashExpertReadRange> ranges;
    for (size_t begin = 0; begin < ordered.size();) {
        size_t end = begin + 1;
        while (end < ordered.size() && end - begin < flash_expert_read_width &&
               ordered[end].expert == ordered[end - 1].expert + 1) {
            ++end;
        }
        ranges.push_back({begin, end});
        begin = end;
    }
    return ranges;
}

// Each source span is contiguous. Its destination slots need not be.
template <class Request, class Read>
void flash_read_expert_family(const Request * requests, size_t count, size_t offset,
                              size_t bytes, unsigned char * slots, Read read) {
    iovec spans[flash_expert_read_width];
    for (size_t i = 0; i < count; ++i) {
        spans[i] = {slots + size_t(requests[i].slot) * bytes, bytes};
    }
    read(offset + size_t(requests[0].expert) * bytes, spans, int(count));
}

// One wave at a time, owned by the context's serialized decode caller.
// The caller participates, and no wave returns while a worker can write to its slots.
class FlashExpertReadPool {
public:
    ~FlashExpertReadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto & thread : workers_) thread.join();
    }

    void run(int count, int parallelism, const std::function<void(int)> & task) {
        const int limit = std::min(count, parallelism);
        if (limit <= 1) {
            for (int i = 0; i < count; ++i) task(i);
            return;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        while (int(workers_.size()) + 1 < limit) workers_.emplace_back([this] { work(); });
        task_ = &task;
        limit_ = limit;
        next_ = 0;
        end_ = count;
        error_ = nullptr;
        ready_.notify_all();
        while (available()) execute(lock);
        finished_.wait(lock, [this] { return next_ == end_ && active_ == 0; });
        task_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:
    bool available() const { return next_ < end_ && active_ < limit_; }

    void work() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [this] { return stopping_ || available(); });
            if (stopping_) return;
            execute(lock);
        }
    }

    void execute(std::unique_lock<std::mutex> & lock) {
        const int index = next_++;
        ++active_;
        lock.unlock();
        std::exception_ptr error;
        try {
            (*task_)(index);
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        if (error && !error_) error_ = error;
        --active_;
        if (next_ == end_ && active_ == 0) finished_.notify_one();
    }

    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::vector<std::thread> workers_;
    const std::function<void(int)> * task_ = nullptr;
    std::exception_ptr error_;
    int limit_ = 0, next_ = 0, end_ = 0, active_ = 0;
    bool stopping_ = false;
};
