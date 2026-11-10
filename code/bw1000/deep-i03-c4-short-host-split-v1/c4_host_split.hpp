#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace deep_c4_host_split {
constexpr unsigned marks = 10;
constexpr unsigned devices = 8;
constexpr unsigned layers = 43;

struct Record {
    std::array<std::int64_t, marks> ns{};
    std::size_t past = 0;
    double original_frame_start = 0;
    unsigned next = 0;
    bool complete = false;
    bool valid = false;
};

inline std::array<std::array<Record, layers>, devices> records{};
inline thread_local Record *active = nullptr;

inline void mark(unsigned index) noexcept {
    if (!active || !active->valid) return;
    if (index != active->next || index >= marks) {
        active->valid = false;
        return;
    }
    using namespace std::chrono;
    const auto now = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    if (index && now < active->ns[index - 1]) active->valid = false;
    active->ns[index] = now;
    ++active->next;
}

class Capture {
public:
    Capture(bool enabled, unsigned device, unsigned layer, std::size_t past,
            double original_frame_start) noexcept {
        if (!enabled || device >= devices || layer >= layers) return;
        if (active) {
            active->valid = false;
            return;
        }
        record_ = &records[device][layer];
        *record_ = Record{};
        record_->past = past;
        record_->original_frame_start = original_frame_start;
        record_->valid = true;
        active = record_;
        mark(0);
    }

    Capture(const Capture &) = delete;
    Capture &operator=(const Capture &) = delete;

    ~Capture() noexcept {
        if (!record_) return;
        record_->complete = record_->valid && record_->next == marks;
        active = nullptr;
    }

private:
    Record *record_ = nullptr;
};
}
