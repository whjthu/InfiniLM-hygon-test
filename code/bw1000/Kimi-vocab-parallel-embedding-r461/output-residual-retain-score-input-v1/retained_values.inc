constexpr unsigned values_per_lane = width / lanes;
static_assert(values_per_lane == 28);

struct RetainedValues {
    std::uint16_t words[values_per_lane];
};

__device__ inline float retain_and_square(const std::uint16_t* row, unsigned lane, RetainedValues& held) {
    float sum = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (unsigned sample = 0; sample < values_per_lane; ++sample) {
        const auto word = row[lane + sample * lanes];
        held.words[sample] = word;
        const float value = decode(word);
        sum = add(sum, mul(value, value));
    }
    return sum;
}

__device__ inline float dot_retained(const RetainedValues& held, const std::uint16_t* norm,
                                     const std::uint16_t* projection, float inv_rms, unsigned lane) {
    float sum = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (unsigned sample = 0; sample < values_per_lane; ++sample) {
        const unsigned column = lane + sample * lanes;
        const auto normalized = encode(mul(mul(decode(held.words[sample]), inv_rms), decode(norm[column])));
        sum = add(sum, mul(decode(normalized), decode(projection[column])));
    }
    return sum;
}
