#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace deep_csa_commit_trace {

enum class Phase { idle, armed, recording, finished, invalid };

// The receipt reports successful host calls and the original stream drain.
// It does not time kernels or infer device event counts from source geometry.
struct Receipt {
    std::uint64_t tag = 0;
    std::uint64_t tokens = 0;
    std::uint64_t requests = 0;
    std::uint64_t commit_rows = 0;
    std::uint64_t direct_aliases = 0;
    std::uint64_t index_select_submissions = 0;
    std::uint64_t store_calls_returned = 0;
    std::uint64_t original_drain_returned = 0;
};

struct State {
    Phase phase = Phase::idle;
    const void* owner = nullptr;
    Receipt receipt;
};

inline thread_local State state;
inline constexpr std::size_t width = 8;

inline int arm(const void* owner, std::uint64_t tag) noexcept {
    if (owner == nullptr || tag == 0 || state.phase != Phase::idle) return 1;
    state = State{Phase::armed, owner, Receipt{}};
    state.receipt.tag = tag;
    return 0;
}

inline int clear(const void* owner) noexcept {
    if (owner == nullptr || owner != state.owner || state.phase == Phase::recording) return 1;
    state = State{};
    return 0;
}

class Forward {
public:
    Forward(const void* owner, std::size_t tokens, std::size_t requests) noexcept {
        if (owner == nullptr || owner != state.owner) return;
        if (state.phase != Phase::armed) {
            state.phase = Phase::invalid;
            return;
        }
        active_ = true;
        state.phase = Phase::recording;
        state.receipt.tokens = tokens;
        state.receipt.requests = requests;
    }

    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    ~Forward() noexcept {
        if (recording()) state.phase = Phase::finished;
    }

    void direct(std::size_t rows) noexcept {
        if (!recording()) return;
        state.receipt.commit_rows = rows;
        ++state.receipt.direct_aliases;
    }

    void index_select(std::size_t rows) noexcept {
        if (!recording()) return;
        state.receipt.commit_rows = rows;
        ++state.receipt.index_select_submissions;
    }

    void store() noexcept {
        if (recording()) ++state.receipt.store_calls_returned;
    }

    void complete_after_drain() noexcept {
        if (recording()) state.receipt.original_drain_returned = 1;
    }

private:
    bool recording() const noexcept { return active_ && state.phase == Phase::recording; }
    bool active_ = false;
};

inline int read(const void* owner, std::uint64_t* out, std::size_t size) noexcept {
    if (owner == nullptr || owner != state.owner || out == nullptr || size != width || state.phase != Phase::finished) return 1;
    const auto& row = state.receipt;
    const std::array<std::uint64_t, width> fields{row.tag, row.tokens, row.requests, row.commit_rows,
        row.direct_aliases, row.index_select_submissions, row.store_calls_returned, row.original_drain_returned};
    for (std::size_t index = 0; index < fields.size(); ++index) out[index] = fields[index];
    state = State{};
    return 0;
}

} // namespace deep_csa_commit_trace
