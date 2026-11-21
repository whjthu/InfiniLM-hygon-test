#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define KIMI_SCORE_DEVICE __device__
#else
#include <cstring>
#define KIMI_SCORE_DEVICE
#endif

namespace kimi_output_residual_scores_v1 {
inline constexpr unsigned width = 7168;
inline constexpr unsigned rows = 512 * 9;
inline constexpr unsigned lanes = 256;
static_assert(width % lanes == 0);

KIMI_SCORE_DEVICE inline float decode(std::uint16_t word) {
    const std::uint32_t bits = std::uint32_t(word) << 16;
#if defined(__CUDACC__) || defined(__HIPCC__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

KIMI_SCORE_DEVICE inline std::uint16_t encode(float value) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    const std::uint32_t bits = __float_as_uint(value);
#else
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
#endif
    if ((bits & 0x7fffffffU) > 0x7f800000U) return std::uint16_t((bits >> 16) | 0x0040U);
    return std::uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

KIMI_SCORE_DEVICE inline float add(float a, float b) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return __fadd_rn(a, b);
#else
    volatile float result = a + b;
    return result;
#endif
}

KIMI_SCORE_DEVICE inline float mul(float a, float b) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return __fmul_rn(a, b);
#else
    volatile float result = a * b;
    return result;
#endif
}

KIMI_SCORE_DEVICE inline float square_lane(const std::uint16_t* values, unsigned lane) {
    float sum = 0;
    for (unsigned col = lane; col < width; col += lanes) {
        const float x = decode(values[col]);
        sum = add(sum, mul(x, x));
    }
    return sum;
}

KIMI_SCORE_DEVICE inline float dot_lane(const std::uint16_t* values, const std::uint16_t* norm,
                                       const std::uint16_t* projection, float inv_rms, unsigned lane) {
    float sum = 0;
    for (unsigned col = lane; col < width; col += lanes) {
        const auto normalized = encode(mul(mul(decode(values[col]), inv_rms), decode(norm[col])));
        sum = add(sum, mul(decode(normalized), decode(projection[col])));
    }
    return sum;
}
}

#undef KIMI_SCORE_DEVICE
