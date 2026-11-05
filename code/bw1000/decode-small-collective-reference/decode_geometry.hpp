#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace decode_ccl {
inline bool valid(int ranks, std::size_t bytes, int dtype) {
    if (ranks == 8) {
        if (dtype == 9) return bytes == 8192 || bytes == 12288;
        if (dtype == 10)
            return bytes == 2048 || bytes == 6824 || bytes == 8192 ||
                   bytes == 8496 || bytes == 8576 || bytes == 16384 || bytes == 24576 || bytes == 28176;
    } else if (ranks == 16 && dtype == 9) {
        return bytes == 7168 || bytes == 14336;
    }
    return false;
}

inline std::size_t width(int dtype) {
    if (dtype == 9) return 2;
    if (dtype == 10) return 4;
    throw std::invalid_argument("only BF16 and F32");
}

inline std::uint32_t value_bits(float value, int dtype) {
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    return width(dtype) == 2 ? bits >> 16 : bits;
}

inline std::vector<unsigned char> seed(int ranks, int rank, std::size_t bytes, int dtype) {
    if (!valid(ranks, bytes, dtype) || rank < 0 || rank >= ranks)
        throw std::invalid_argument("decode collective shape or rank");
    const auto bytes_per_element = width(dtype);
    const auto bits = value_bits(float(rank + 1), dtype);
    std::vector<unsigned char> out(bytes);
    for (std::size_t pos = 0; pos < bytes; pos += bytes_per_element)
        std::memcpy(out.data() + pos, &bits, bytes_per_element);
    return out;
}

inline std::size_t wrong_count(const std::vector<unsigned char>& output, int ranks, int dtype) {
    if (!valid(ranks, output.size(), dtype))
        throw std::invalid_argument("decode collective output shape");
    const auto bytes_per_element = width(dtype);
    const auto expected = value_bits(float(ranks * (ranks + 1) / 2), dtype);
    std::size_t wrong = 0;
    for (std::size_t pos = 0; pos < output.size(); pos += bytes_per_element) {
        std::uint32_t bits{};
        std::memcpy(&bits, output.data() + pos, bytes_per_element);
        wrong += bits != expected;
    }
    return wrong;
}
}
