#include "local_runtime.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

extern "C" int deep_inverse_rope_h1_v1(void*, const void*, const void*, size_t, size_t, void*);
extern "C" int deep_inverse_rope_h2_v1(void*, const void*, const void*, size_t, size_t, void*);

using Arm = decltype(&deep_inverse_rope_h1_v1);
constexpr std::array<Arm, 2> arms{deep_inverse_rope_h1_v1, deep_inverse_rope_h2_v1};

float from_bits(uint16_t input) {
    return __uint_as_float(uint32_t(input) << 16);
}

// Separate rounding implementation and explicit standalone FP32 products for
// the in-place original SDK equations, after the original prefix copy.
uint16_t rounded(float x) {
    const uint32_t raw = __float_as_uint(x);
    const uint16_t top = static_cast<uint16_t>(raw >> 16);
    const uint16_t low = static_cast<uint16_t>(raw);
    return static_cast<uint16_t>(top + (low > 0x8000 || (low == 0x8000 && top % 2)));
}

void original(std::vector<uint16_t>& out, const float* table, size_t heads, size_t tokens) {
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            uint16_t* row = out.data() + (t * heads + h) * 512;
            for (size_t j = 0; j < 32; ++j) {
                const float x = from_bits(row[448 + 2 * j]);
                const float y = from_bits(row[449 + 2 * j]);
                const float c = table[t * 64 + j];
                const float s = -table[t * 64 + 32 + j];
                volatile float xc = x * c, ys = y * s, yc = y * c, xs = x * s;
                row[448 + 2 * j] = rounded(xc - ys);
                row[449 + 2 * j] = rounded(yc + xs);
            }
        }
    }
}

int main() {
    int stream_value = 0;
    const auto stream = static_cast<void*>(&stream_value);
    size_t sizes = 0, cases = 0, prefix = 0, rotated = 0, changed = 0, rejects = 0;
    const std::array<std::array<size_t, 2>, 6> shapes{{
        {1, 8}, {17, 8}, {128, 8}, {512, 8}, {1, 64}, {17, 64}
    }};
    for (const auto& shape : shapes) {
        const auto tokens = shape[0];
        const auto heads = shape[1];
        const auto count = tokens * heads * 512;
        std::vector<uint16_t> storage(count + 256, 0x6a6a);
        uint16_t* const in = storage.data() + 128;
        for (size_t row = 0; row < count / 512; ++row) {
            for (size_t col = 0; col < 512; ++col) {
                const size_t i = row * 512 + col;
                if (col < 448) {
                    in[i] = static_cast<uint16_t>(i * 413 + row * 17);
                } else {
                    in[i] = rounded(float(int((i * 109 + row * 31) % 1021) - 510) / 64.f);
                    if (i % 17 == 0) {
                        in[i] = (i % 2) ? 0x8000 : 0;
                    }
                }
            }
        }
        const auto initial = storage;
        for (const size_t start : {size_t{0}, size_t{8496}, size_t{32768} - tokens}) {
            std::vector<float> table(tokens * 64 + 128, 123.f);
            float* const data = table.data() + 64;
            for (size_t t = 0; t < tokens; ++t) {
                for (size_t j = 0; j < 32; ++j) {
                    const double position = double(start + t);
                    const double a = position * std::pow(160000., -2. * double(j) / 64.);
                    data[t * 64 + j] = float(std::cos(a));
                    data[t * 64 + j + 32] = float(std::sin(a));
                }
            }
            const auto saved_table = table;
            std::vector<uint16_t> expected(in, in + count);
            original(expected, data, heads, tokens);
            for (size_t candidate = 0; candidate < arms.size(); ++candidate) {
                std::vector<uint16_t> guarded(count + 256, 0x55aa);
                auto before = launch_count;
                uint16_t* const output = guarded.data() + 128;
                assert(arms[candidate](output, in, data, tokens, heads, stream) == 0);
                assert(launch_count == before + 1 && last_stream == stream);
                assert(last_grid == tokens * heads / (candidate + 1));
                assert(last_threads == 64 * (candidate + 1));
                assert(storage == initial && table == saved_table);
                for (size_t i = 0; i < 128; ++i) {
                    assert(guarded[i] == 0x55aa && guarded[128 + count + i] == 0x55aa);
                }
                for (size_t i = 0; i < count; ++i) {
                    assert(output[i] == expected[i]);
                    if (i % 512 < 448) {
                        assert(output[i] == in[i]);
                        ++prefix;
                    } else {
                        ++rotated;
                        changed += output[i] != in[i];
                    }
                }
                ++cases;
            }
        }
        ++sizes;
    }

    std::array<uint16_t, 8193> in{}, out{};
    std::array<float, 128> angles{};
    for (const auto arm : arms) {
        const unsigned before = launch_count;
        assert(arm(nullptr, in.data(), angles.data(), 1, 8, stream) == 1);
        assert(arm(out.data(), nullptr, angles.data(), 1, 8, stream) == 1);
        assert(arm(out.data(), in.data(), nullptr, 1, 8, stream) == 1);
        assert(arm(out.data(), in.data(), angles.data(), 1, 8, nullptr) == 1);
        assert(arm(out.data(), in.data(), angles.data(), 0, 8, stream) == 1);
        assert(arm(out.data(), in.data(), angles.data(), 513, 8, stream) == 1);
        assert(arm(out.data(), in.data(), angles.data(), 1, 16, stream) == 1);
        assert(arm(in.data(), in.data(), angles.data(), 1, 8, stream) == 1);
        assert(arm(in.data() + 1, in.data(), angles.data(), 1, 8, stream) == 1);
        assert(arm(out.data(), in.data(), reinterpret_cast<const float*>(in.data()), 1, 8, stream) == 1);
        assert(arm(reinterpret_cast<uint8_t*>(out.data()) + 1, in.data(), angles.data(), 1, 8, stream) == 1);
        assert(launch_count == before);
        rejects += 11;
        injected_error = 73;
        assert(arm(out.data(), in.data(), angles.data(), 1, 8, stream) == 73);
        assert(launch_count == before + 1);
        injected_error = 0;
    }
    assert(changed > 0);
    std::cout << "{\"shapes\":" << sizes << ",\"cases\":" << cases
              << ",\"prefix_exact\":" << prefix << ",\"rotated_exact\":" << rotated
              << ",\"changed_tail_values\":" << changed << ",\"invalid_no_launch\":" << rejects
              << ",\"launch_error_propagations\":2}\n";
}
