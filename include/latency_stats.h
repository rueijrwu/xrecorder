#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

// Lightweight ring-buffer latency tracker. Percentiles are computed on
// demand (telemetry is polled at ~10 Hz, so a sort-on-read is cheap) rather
// than maintained incrementally.
class LatencyStats {
public:
    explicit LatencyStats(size_t capacity = 2048) : capacity_(capacity) {
        samples_.reserve(capacity);
    }

    void Record(double value_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.size() < capacity_) {
            samples_.push_back(value_us);
        } else {
            samples_[write_idx_] = value_us;
        }
        write_idx_ = (write_idx_ + 1) % capacity_;
        count_.fetch_add(1, std::memory_order_relaxed);
        double prev_max = max_.load(std::memory_order_relaxed);
        while (value_us > prev_max && !max_.compare_exchange_weak(prev_max, value_us)) {}
    }

    struct Snapshot {
        double mean = 0.0;
        double p95 = 0.0;
        double p99 = 0.0;
        double p999 = 0.0;
        double max = 0.0;
        uint64_t total_count = 0;
    };

    Snapshot GetSnapshot() const {
        std::vector<double> sorted;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            sorted = samples_;
        }
        Snapshot s;
        s.total_count = count_.load(std::memory_order_relaxed);
        s.max = max_.load(std::memory_order_relaxed);
        if (sorted.empty()) return s;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0.0;
        for (double v : sorted) sum += v;
        s.mean = sum / static_cast<double>(sorted.size());
        auto pct = [&](double p) {
            size_t idx = static_cast<size_t>(p * static_cast<double>(sorted.size() - 1));
            return sorted[std::min(idx, sorted.size() - 1)];
        };
        s.p95 = pct(0.95);
        s.p99 = pct(0.99);
        s.p999 = pct(0.999);
        return s;
    }

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::vector<double> samples_;
    size_t write_idx_ = 0;
    std::atomic<uint64_t> count_{0};
    std::atomic<double> max_{0.0};
};
