#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>

namespace glm_projection_host_timing {

using Clock = std::chrono::steady_clock;

struct Sample {
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t tokens = 0;
    std::uint64_t cached = 0;
    std::uint64_t lease_active = 0;
    std::uint64_t fence_ns = 0;
    std::uint64_t return_cleanup_ns = 0;
    std::uint64_t next_quant_gap_ns = 0;
    std::uint64_t next_quant_observed = 0;
};

struct Capture {
    static constexpr std::size_t capacity = 32;
    const void* owner = nullptr;
    std::array<Sample, capacity> samples{};
    std::size_t size = 0;
    bool waiting_for_quant = false;
    Clock::time_point last_return{};
    bool dropped = false;

    bool begin(const void* runtime) noexcept {
        if (!runtime || owner) return false;
        size = 0;
        waiting_for_quant = false;
        dropped = false;
        owner = runtime;
        return true;
    }

    void before_quant(const void* runtime, Clock::time_point now) noexcept {
        if (runtime != owner || !waiting_for_quant) return;
        auto& previous = samples[size - 1];
        previous.next_quant_gap_ns = elapsed(last_return, now);
        previous.next_quant_observed = 1;
        waiting_for_quant = false;
    }

    void finish_sample(const void* runtime, const Sample& sample, Clock::time_point now) noexcept {
        if (runtime != owner) return;
        if (size == capacity) {
            dropped = true;
            return;
        }
        samples[size++] = sample;
        last_return = now;
        waiting_for_quant = true;
    }

    bool end(const void* runtime) noexcept {
        if (!owner || runtime != owner) return false;
        owner = nullptr;
        waiting_for_quant = false;
        return true;
    }

    static std::uint64_t elapsed(Clock::time_point start, Clock::time_point end) noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

inline thread_local Capture capture;

class ProjectionTiming {
public:
    ProjectionTiming(const void* runtime, std::size_t rows, std::size_t columns, std::size_t tokens) noexcept
        : runtime_(capture.owner == runtime ? runtime : nullptr), sample_{rows, columns, tokens} {}

    ProjectionTiming(const ProjectionTiming&) = delete;
    ProjectionTiming& operator=(const ProjectionTiming&) = delete;

    bool enabled() const noexcept { return runtime_ != nullptr; }

    void before_quant() noexcept {
        if (enabled() && capture.waiting_for_quant) capture.before_quant(runtime_, Clock::now());
    }

    void before_fence(bool cached, bool lease_active) noexcept {
        sample_.cached = cached;
        sample_.lease_active = lease_active;
        before_fence_ = Clock::now();
    }

    void after_fence() noexcept {
        if (!enabled()) return;
        after_fence_ = Clock::now();
        sample_.fence_ns = Capture::elapsed(before_fence_, after_fence_);
        fence_completed_ = true;
    }

    ~ProjectionTiming() noexcept {
        if (!fence_completed_ || std::uncaught_exceptions() != exceptions_) return;
        const auto now = Clock::now();
        sample_.return_cleanup_ns = Capture::elapsed(after_fence_, now);
        capture.finish_sample(runtime_, sample_, now);
    }

private:
    const void* runtime_;
    Sample sample_;
    int exceptions_ = std::uncaught_exceptions();
    bool fence_completed_ = false;
    Clock::time_point before_fence_{};
    Clock::time_point after_fence_{};
};

}  // namespace glm_projection_host_timing
