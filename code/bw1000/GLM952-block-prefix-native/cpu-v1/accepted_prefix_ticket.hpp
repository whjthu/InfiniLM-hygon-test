#pragma once

#include "accepted_prefix_lengths.hpp"
#include <cstdint>

namespace infinilm::models::glm_moe_dsa::block_verify {

class PrefixTicket {
public:
    bool unresolved() const noexcept { return state_ != State::Idle; }

    bool can_stage(std::uint64_t sequence, std::size_t past, std::size_t width) const noexcept {
        return !unresolved() && (!used_ || sequence > sequence_) &&
            (width == 4 || width == 8) && past <= 32768 && width <= 32768 - past;
    }

    // Called only after the original forward returns, including its owner drain.
    void record_forward(std::uint64_t sequence, std::size_t past, std::size_t width,
                        bool completed) noexcept {
        used_ = true;
        sequence_ = sequence;
        past_ = past;
        width_ = width;
        state_ = completed ? State::Staged : State::Failed;
    }

    bool can_prepare(std::uint64_t sequence, StagedPrefix prefix) const noexcept {
        return state_ == State::Staged && matches(sequence, prefix) && prefix.accepted <= width_;
    }

    void prepared(std::size_t accepted) noexcept {
        accepted_ = accepted;
        state_ = State::Prepared;
    }

    bool take_publish(std::uint64_t sequence, StagedPrefix prefix) noexcept {
        if (state_ != State::Prepared || !matches(sequence, prefix) || prefix.accepted != accepted_) {
            return false;
        }
        // If the write throws or is rejected, only a successful full reset
        // can reopen this Context. Never retry an ambiguous publish.
        state_ = State::Failed;
        return true;
    }

    void published() noexcept { state_ = State::Idle; }
    void reset_completed() noexcept { state_ = State::Idle; }

private:
    enum class State { Idle, Staged, Prepared, Failed };

    bool matches(std::uint64_t sequence, StagedPrefix prefix) const noexcept {
        return sequence == sequence_ && prefix.past == past_ && prefix.width == width_;
    }

    bool used_ = false;
    State state_ = State::Idle;
    std::uint64_t sequence_ = 0;
    std::size_t past_ = 0, width_ = 0, accepted_ = 0;
};

}  // namespace infinilm::models::glm_moe_dsa::block_verify
