#pragma once
// Isolated candidate only: not included in the shipping native runtime.
// A serialized owner must await run() before destroying this pool. The caller
// participates, so N concurrent readers need only N-1 persistent helper threads.
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace flash_local {
class bounded_io_pool {
public:
    explicit bounded_io_pool(unsigned readers) : readers_(readers) {
        if (readers < 1 || readers > 8) throw std::invalid_argument("I/O readers must be 1...8");
        try {
            helpers_.reserve(readers - 1);
            for (unsigned index = 1; index < readers; ++index) helpers_.emplace_back([this] { worker(); });
        } catch (...) { stop_and_join(); throw; }
    }
    bounded_io_pool(const bounded_io_pool &) = delete;
    bounded_io_pool & operator=(const bounded_io_pool &) = delete;
    ~bounded_io_pool() { stop_and_join(); }

    void run(std::size_t jobs, std::function<void(std::size_t)> operation) {
        if (jobs > 256 || !operation) throw std::invalid_argument("Invalid bounded I/O wave");
        std::unique_lock<std::mutex> lock(mutex_);
        if (active_ || stopping_) throw std::logic_error("Overlapping I/O wave");
        if (jobs == 0) return;
        operation_ = std::move(operation);
        count_ = jobs; next_.store(0, std::memory_order_relaxed);
        error_ = nullptr; remaining_ = helpers_.size(); active_ = true; ++generation_;
        wake_.notify_all(); lock.unlock();
        consume();
        lock.lock();
        done_.wait(lock, [this] { return remaining_ == 0; });
        auto failure = error_;
        operation_ = nullptr; active_ = false; ++completed_waves_; completed_jobs_ += jobs;
        lock.unlock();
        if (failure) std::rethrow_exception(failure);
    }
    unsigned readers() const { return readers_; }
    unsigned thread_launches() const { return readers_ - 1; }
    std::uint64_t completed_waves() const {
        std::lock_guard<std::mutex> lock(mutex_); return completed_waves_;
    }
    std::uint64_t completed_jobs() const {
        std::lock_guard<std::mutex> lock(mutex_); return completed_jobs_;
    }
    static std::size_t live_helpers() { return live_helpers_.load(); }
private:
    void consume() {
        for (;;) {
            const auto index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_) break;
            try { operation_(index); }
            catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
            }
        }
    }
    void worker() {
        live_helpers_.fetch_add(1);
        std::uint64_t observed = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            wake_.wait(lock, [this, observed] { return stopping_ || generation_ != observed; });
            if (stopping_) break;
            observed = generation_; lock.unlock(); consume(); lock.lock();
            if (--remaining_ == 0) done_.notify_one();
        }
        live_helpers_.fetch_sub(1);
    }
    void stop_and_join() noexcept {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        wake_.notify_all();
        for (auto & helper : helpers_) if (helper.joinable()) helper.join();
    }
    const unsigned readers_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::vector<std::thread> helpers_;
    std::function<void(std::size_t)> operation_;
    std::atomic<std::size_t> next_{0};
    std::size_t count_ = 0, remaining_ = 0;
    std::uint64_t generation_ = 0, completed_waves_ = 0, completed_jobs_ = 0;
    bool active_ = false, stopping_ = false;
    std::exception_ptr error_;
    inline static std::atomic<std::size_t> live_helpers_{0};
};
} // namespace flash_local
