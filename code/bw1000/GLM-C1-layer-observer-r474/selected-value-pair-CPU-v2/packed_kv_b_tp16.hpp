#pragma once

#include "packed_kv_b.hpp"

namespace glm_nov_value::tp16 {
inline constexpr std::size_t ranks = 16, heads = 4, original_rows = 1792;
inline constexpr std::size_t original_scale_rows = ranks * original_rows / block;
inline constexpr std::array<std::size_t, 10> tiles{1, 2, 3, 5, 6, 8, 9, 10, 12, 13};
inline constexpr std::array<std::size_t, heads> starts{64, 384, 704, 1024};
inline constexpr std::size_t packed_rows = tiles.size() * block;

inline Packed pack(const std::uint8_t* weight, std::size_t weight_bytes,
                   const float* global_scale, std::size_t scale_count, std::size_t rank) {
    if (!weight || !global_scale || weight_bytes != original_rows * columns ||
        scale_count != original_scale_rows * scale_columns || rank >= ranks)
        throw std::invalid_argument("GLM TP16 kv_b weight/scale geometry");

    Packed result{std::vector<std::uint8_t>(packed_rows * columns),
                  std::vector<float>(tiles.size() * scale_columns)};
    const auto rank_scale_base = rank * (original_rows / block);
    for (std::size_t destination = 0; destination < tiles.size(); ++destination) {
        const auto source = tiles[destination];
        std::copy_n(weight + source * block * columns, block * columns,
                    result.weight.data() + destination * block * columns);
        std::memcpy(result.scale.data() + destination * scale_columns,
                    global_scale + (rank_scale_base + source) * scale_columns,
                    scale_columns * sizeof(float));
    }
    return result;
}
}
