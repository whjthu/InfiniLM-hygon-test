#pragma once
#include <array>
#include <cstddef>

namespace glm_r474_depth_detail {

struct Pair {
    unsigned ordinal;
    std::size_t full;
    std::size_t shared;

    constexpr bool accepts(std::size_t layer, bool full_indexer) const {
        return layer == (full_indexer ? full : shared);
    }
};

inline constexpr std::array<Pair, 3> kPairs{{{1, 18, 19}, {2, 38, 40}, {3, 66, 69}}};

constexpr const Pair* select(bool marked_c1, unsigned ordinal) {
    return marked_c1 && ordinal >= 1 && ordinal <= kPairs.size()
        ? &kPairs[ordinal - 1] : nullptr;
}

}
