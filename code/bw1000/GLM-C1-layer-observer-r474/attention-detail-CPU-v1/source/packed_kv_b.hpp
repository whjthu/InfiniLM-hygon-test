#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace glm_nov_value {
inline constexpr std::size_t ranks = 8, block = 128, columns = 512;
inline constexpr std::size_t heads = 8, value_columns = 256, original_rows = 3584;
inline constexpr std::size_t scale_columns = columns / block;
inline constexpr std::size_t original_scale_rows = ranks * original_rows / block;
inline constexpr std::array<std::size_t, 20> tiles{
    1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19, 20, 22, 23, 24, 26, 27};
inline constexpr std::array<std::size_t, heads> starts{
    64, 384, 704, 1024, 1344, 1664, 1984, 2304};
inline constexpr std::size_t packed_rows = tiles.size() * block;

struct Packed {
    std::vector<std::uint8_t> weight;
    std::vector<float> scale;
};

// The retained TP8 provider receives a rank-local FP8 byte tensor and an
// unsharded scale tensor. Copy whole original blocks and their scale bytes.
inline Packed pack(const std::uint8_t* weight, std::size_t weight_bytes,
                   const float* global_scale, std::size_t scale_count, std::size_t rank) {
    static_assert(sizeof(float) == 4, "FP32 scale bytes");
    if (!weight || !global_scale || weight_bytes != original_rows * columns ||
        scale_count != original_scale_rows * scale_columns || rank >= ranks)
        throw std::invalid_argument("GLM original kv_b weight/scale geometry");
    Packed output{std::vector<std::uint8_t>(packed_rows * columns),
                  std::vector<float>(tiles.size() * scale_columns)};
    const auto rank_scale_base = rank * (original_rows / block);
    for (std::size_t packed = 0; packed < tiles.size(); ++packed) {
        const auto original = tiles[packed];
        std::copy_n(weight + original * block * columns, block * columns,
                    output.weight.data() + packed * block * columns);
        std::memcpy(output.scale.data() + packed * scale_columns,
                    global_scale + (rank_scale_base + original) * scale_columns,
                    scale_columns * sizeof(float));
    }
    return output;
}
}
