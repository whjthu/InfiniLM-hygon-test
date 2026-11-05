#pragma once
#include "../decode_geometry.hpp"
#include <array>

namespace decode_ccl {
enum class Phase { index_score, attention_output, moe };
struct ChainPoint {
    int layer;
    Phase phase;
    int dtype;
    std::size_t bytes;
};

inline std::array<ChainPoint, 107> deep_chain(std::size_t past) {
    if (past != 8496 && past != 8575)
        throw std::invalid_argument("only the two r467 decode windows");
    constexpr std::array<int,43> ratios{0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4};
    std::array<ChainPoint,107> points{};
    std::size_t next = 0;
    for (int layer = 0; layer < static_cast<int>(ratios.size()); ++layer) {
        if (ratios[layer] == 4)
            points.at(next++) = {layer, Phase::index_score, 10, 4 * ((past + 1) / 4)};
        points.at(next++) = {layer, Phase::attention_output, 9, 8192};
        points.at(next++) = {layer, Phase::moe, 10, 16384};
    }
    if (next != points.size()) throw std::logic_error("wrong layer schedule");
    return points;
}

// Every slot has a distinct element marker. All BF16 intermediate sums are small integers.
inline std::vector<unsigned char> chain_values(const ChainPoint& point, std::size_t slot,
                                               int rank, bool reduced) {
    if (!valid(8, point.bytes, point.dtype) || rank < 0 || rank >= 8 || slot >= 107)
        throw std::invalid_argument("invalid chain buffer geometry");
    const auto step = width(point.dtype);
    const auto ordinary = value_bits(reduced ? 36.f : float(rank + 1), point.dtype);
    const auto marker = value_bits(reduced ? 44.f : float(rank + 2), point.dtype);
    std::vector<unsigned char> bytes(point.bytes);
    for (std::size_t i = 0; i < bytes.size() / step; ++i)
        std::memcpy(bytes.data() + i * step, i == slot ? &marker : &ordinary, step);
    return bytes;
}

// The caller owns one buffer per slot on one owner stream. Seed, validation,
// and allocation are outside the timed section. No Model or PP transfer.
template <class Ops>
void chain_once(Ops& ops, const std::array<ChainPoint,107>& points, bool fence_each) {
    for (std::size_t slot = 0; slot < points.size(); ++slot) ops.seed(slot, points[slot]);
    ops.fence();
    ops.begin_timing();
    for (std::size_t slot = 0; slot < points.size(); ++slot) {
        ops.allreduce_in_place(slot, points[slot]);
        if (fence_each && slot + 1 < points.size()) ops.fence();
    }
    ops.end_enqueue();
    ops.fence();
    ops.end_timing();
    for (std::size_t slot = 0; slot < points.size(); ++slot) ops.validate(slot, points[slot]);
}
}
