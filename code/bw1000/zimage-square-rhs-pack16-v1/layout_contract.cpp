#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t M = 288, N = 3840, K = 3840, Guard = 256;
constexpr std::uint16_t Canary = 0xc11c;

std::size_t physical_transposed(std::size_t k, std::size_t n) { return k + K * n; }
std::size_t physical_packed(std::size_t k, std::size_t n) { return N * k + n; }

void check(bool ok) {
    if (!ok) throw std::runtime_error("host layout contract failed");
}

int code(std::size_t k, std::size_t n) {
    return static_cast<int>((11 * k + 17 * n + 7 * (k ^ n)) % 16);
}

// All entries are exactly representable multiples of 1/32 with magnitude <= 1/4.
constexpr std::array<std::uint16_t, 16> Bits = {
    0xbe80, 0xbe60, 0xbe40, 0xbe20, 0xbe00, 0xbdc0, 0xbd80, 0xbd00,
    0x0000, 0x3d00, 0x3d80, 0x3dc0, 0x3e00, 0x3e20, 0x3e40, 0x3e60
};

int decode(std::uint16_t bits) {
    for (std::size_t i = 0; i < Bits.size(); ++i)
        if (Bits[i] == bits) return static_cast<int>(i) - 8;
    throw std::runtime_error("unexpected BF16 payload");
}
} // namespace

int main() {
    std::vector<std::uint16_t> transposed(K * N + 2 * Guard, Canary), packed(K * N + 2 * Guard, Canary);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t n = 0; n < N; ++n)
            transposed[Guard + physical_transposed(k, n)] = Bits[code(k, n)];
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t n = 0; n < N; ++n)
            packed[Guard + physical_packed(k, n)] = transposed[Guard + physical_transposed(k, n)];
    std::size_t changed_layout = 0;
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t n = 0; n < N; ++n) {
            check(packed[Guard + physical_packed(k, n)] == Bits[code(k, n)]);
            check(transposed[Guard + physical_transposed(k, n)] == Bits[code(k, n)]);
            changed_layout += packed[Guard + physical_packed(k, n)] != transposed[Guard + physical_packed(k, n)];
        }
    }
    check(changed_layout > K * N / 2);
    for (std::size_t i = 0; i < Guard; ++i) {
        check(packed[i] == Canary && transposed[i] == Canary);
        check(packed[Guard + K * N + i] == Canary && transposed[Guard + K * N + i] == Canary);
    }
    std::size_t dots = 0;
    for (std::size_t row : {std::size_t(0), std::size_t(127), M - 1}) {
        for (std::size_t n : {std::size_t(0), std::size_t(127), N - 1}) {
            int via_transposed = 0, via_packed = 0, expected = 0;
            for (std::size_t k = 0; k < K; ++k) {
                int lhs = static_cast<int>((5 * row + 13 * k) % 16) - 8;
                via_transposed += lhs * decode(transposed[Guard + physical_transposed(k, n)]);
                via_packed += lhs * decode(packed[Guard + physical_packed(k, n)]);
                expected += lhs * (code(k, n) - 8);
            }
            check(via_packed == expected && via_transposed == expected);
            check(expected > -(1 << 24) && expected < (1 << 24));
            ++dots;
        }
    }
    std::cout << "{\"RHS_BF16_elements\":" << K * N << ",\"physical_offsets_with_different_values\":"
              << changed_layout << ",\"independent_sampled_dot_products\":" << dots
              << ",\"GPU_calls\":0}\n";
}
