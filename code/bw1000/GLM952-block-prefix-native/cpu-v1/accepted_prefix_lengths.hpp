#pragma once

#include <array>
#include <cstddef>

namespace infinilm::models::glm_moe_dsa::block_verify {

struct StagedPrefix {
    std::size_t past;
    std::size_t width;
    std::size_t accepted;
};

template <std::size_t Layers, std::size_t Indexers>
struct OwnerLengths {
    std::size_t* text;
    std::array<std::size_t*, Layers> attention;
    std::array<std::size_t*, Indexers> selector;

    bool can_publish_after_owner_drain(StagedPrefix prefix) const noexcept {
        static_assert((Layers == 38 && Indexers == 11) || (Layers == 40 && Indexers == 10));
        constexpr std::size_t capacity = 32768;
        if ((prefix.width != 4 && prefix.width != 8) || prefix.accepted > prefix.width ||
            prefix.past > capacity || prefix.width > capacity - prefix.past) {
            return false;
        }
        const auto staged_end = prefix.past + prefix.width;
        const auto slots = all_slots();
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (slots[i] == nullptr || *slots[i] != staged_end) return false;
            for (std::size_t j = 0; j < i; ++j) {
                if (slots[i] == slots[j]) return false;
            }
        }
        return true;
    }

    bool publish_after_owner_drain(StagedPrefix prefix) const noexcept {
        // Only the owning thread may call this, after the original forward drain.
        // The distributed caller must handle failure before another forward.
        if (!can_publish_after_owner_drain(prefix)) return false;
        const auto accepted_end = prefix.past + prefix.accepted;
        for (auto* slot : all_slots()) *slot = accepted_end;
        return true;
    }

private:
    auto all_slots() const noexcept {
        std::array<std::size_t*, 1 + Layers + Indexers> slots;
        slots[0] = text;
        for (std::size_t i = 0; i < Layers; ++i) slots[i + 1] = attention[i];
        for (std::size_t i = 0; i < Indexers; ++i) slots[i + 1 + Layers] = selector[i];
        return slots;
    }
};

using FrontOwnerLengths = OwnerLengths<38, 11>;
using TailOwnerLengths = OwnerLengths<40, 10>;

}  // namespace infinilm::models::glm_moe_dsa::block_verify
