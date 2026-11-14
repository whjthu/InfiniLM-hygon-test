#include "packed_kv_b_tp16.hpp"

#include <iostream>

namespace base = glm_nov_value;
namespace tp16 = glm_nov_value::tp16;

int main() {
    constexpr auto width = base::columns;
    constexpr auto block = base::block;
    constexpr auto scale_cols = base::scale_columns;
    std::vector<std::uint8_t> weights(tp16::original_rows * width);
    std::vector<float> scales(tp16::original_scale_rows * scale_cols);
    for (std::size_t i = 0; i < scales.size(); ++i)
        scales[i] = static_cast<float>(i + 1);

    for (std::size_t rank = 0; rank < tp16::ranks; ++rank) {
        for (std::size_t i = 0; i < weights.size(); ++i)
            weights[i] = static_cast<std::uint8_t>((i % width + 23 * (i / width) + rank) % 251);
        const auto packed = tp16::pack(weights.data(), weights.size(), scales.data(), scales.size(), rank);
        if (packed.weight.size() != tp16::packed_rows * width ||
            packed.scale.size() != tp16::tiles.size() * scale_cols)
            return 1;
        for (std::size_t head = 0; head < tp16::heads; ++head) {
            for (std::size_t value = 0; value < base::value_columns; ++value) {
                const std::size_t original = head * 448 + 192 + value;
                const std::size_t destination = tp16::starts[head] + value;
                const auto* src = weights.data() + original * width;
                if (std::memcmp(src, packed.weight.data() + destination * width, width) != 0)
                    return 2;
                for (std::size_t column = 0; column < scale_cols; ++column) {
                    const auto global_row = rank * (tp16::original_rows / block) + original / block;
                    if (scales[global_row * scale_cols + column] !=
                        packed.scale[(destination / block) * scale_cols + column])
                        return 3;
                }
            }
        }
    }

    const auto rejects = [&](const std::uint8_t* data, std::size_t bytes,
                             const float* scale, std::size_t count, std::size_t rank) {
        try {
            tp16::pack(data, bytes, scale, count, rank);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    if (!rejects(nullptr, weights.size(), scales.data(), scales.size(), 0) ||
        !rejects(weights.data(), weights.size() - 1, scales.data(), scales.size(), 0) ||
        !rejects(weights.data(), weights.size(), nullptr, scales.size(), 0) ||
        !rejects(weights.data(), weights.size(), scales.data(), scales.size() - 1, 0) ||
        !rejects(weights.data(), weights.size(), scales.data(), scales.size(), tp16::ranks))
        return 4;

    std::cout << "16 TP16 ranks and five invalid inputs passed\n";
}
