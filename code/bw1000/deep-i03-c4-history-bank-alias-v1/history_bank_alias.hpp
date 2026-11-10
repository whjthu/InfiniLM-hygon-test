#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace deep_c4_history_bank_alias {
inline std::mutex mutex;
inline int mode = 0;
inline std::size_t active = 0;
inline std::array<std::array<std::uint64_t, 4>, 8> counters{};

inline bool eligible(std::size_t tokens, std::size_t requests, std::size_t past,
                     std::size_t compressed_count, bool has_boundary) noexcept {
    return tokens == 1 && requests == 1 && compressed_count > 0 &&
           compressed_count <= 65536 && past / 4 == compressed_count &&
           past % 4 != 3 && !has_boundary;
}

inline int set(int value) {
    if (value != 0 && value != 1) return 1;
    std::lock_guard<std::mutex> lock(mutex);
    if (active) return 2;
    mode = value;
    return 0;
}

inline int read(int device, std::uint64_t *out, std::size_t count) {
    if (device < 0 || device >= 8 || !out || count != 4) return 1;
    std::lock_guard<std::mutex> lock(mutex);
    if (active) return 2;
    for (std::size_t i = 0; i < count; ++i) out[i] = counters[device][i];
    return 0;
}

class Use {
public:
    explicit Use(int device) : device_(device) {
        if (device < 0 || device >= 8) throw std::invalid_argument("C4 history alias device");
        std::lock_guard<std::mutex> lock(mutex);
        enabled = mode == 1;
        ++active;
    }
    Use(const Use &) = delete;
    Use &operator=(const Use &) = delete;

    void complete(bool candidate, bool used, std::size_t copied_bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        auto &row = counters[device_];
        ++row[0];
        row[1] += candidate;
        row[2] += used;
        row[3] += used ? copied_bytes : 0;
    }
    ~Use() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        --active;
    }
    bool enabled = false;

private:
    int device_;
};
}
