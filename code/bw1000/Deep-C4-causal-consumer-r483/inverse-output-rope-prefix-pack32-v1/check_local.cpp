#include "local_runtime.hpp"
#include "independent_reference.inc"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

extern "C" int deep_inverse_rope_h1_pack32_v1(void*, const void*, const void*, size_t, size_t, void*);

void run_case(size_t tokens, size_t heads, size_t position, unsigned input_offset,
              unsigned output_offset, size_t& prefix, size_t& tail) {
    const size_t count = tokens * heads * 512;
    std::vector<uint16_t> storage(count + 260, 0x6a6a), guarded(count + 260, 0x55aa);
    std::vector<float> table(tokens * 64 + 128, 123.f);
    auto* const input = storage.data() + 128 + input_offset;
    auto* const output = guarded.data() + 128 + output_offset;
    auto* const values = table.data() + 64;
    assert(reinterpret_cast<uintptr_t>(input) % 4 == 2 * input_offset);
    assert(reinterpret_cast<uintptr_t>(output) % 4 == 2 * output_offset);
    for (size_t i = 0; i < count; ++i) {
        input[i] = i % 512 < 448 ? static_cast<uint16_t>(i * 413 + (i / 512) * 17)
                                 : rounded(float(int((i * 109) % 1021) - 510) / 64.f);
    }
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t j = 0; j < 32; ++j) {
            const double angle = double(position + t) * std::pow(160000., -2. * double(j) / 64.);
            values[t * 64 + j] = float(std::cos(angle));
            values[t * 64 + 32 + j] = float(std::sin(angle));
        }
    }
    const auto input_saved = storage;
    const auto table_saved = table;
    std::vector<uint16_t> expected(input, input + count);
    original(expected, values, heads, tokens);
    const unsigned previous = launch_count;
    int stream_storage = 0;
    auto* stream = static_cast<void*>(&stream_storage);
    assert(deep_inverse_rope_h1_pack32_v1(output, input, values, tokens, heads, stream) == 0);
    assert(launch_count == previous + 1 && last_stream == stream);
    assert(last_grid == tokens * heads && last_threads == 64);
    assert(last_packed_prefix == (input_offset == 0 && output_offset == 0));
    assert(storage == input_saved && table == table_saved);
    for (size_t i = 0; i < 128 + output_offset; ++i) {
        assert(guarded[i] == 0x55aa);
    }
    for (size_t i = 128 + output_offset + count; i < guarded.size(); ++i) {
        assert(guarded[i] == 0x55aa);
    }
    for (size_t i = 0; i < count; ++i) {
        assert(output[i] == expected[i]);
        if (i % 512 < 448) {
            assert(output[i] == input[i]);
            ++prefix;
        } else {
            ++tail;
        }
    }
}

int main() {
    const std::array<std::array<size_t, 2>, 6> shapes{{{1, 8}, {17, 8}, {128, 8}, {512, 8}, {1, 64}, {17, 64}}};
    size_t prefix = 0, tail = 0, aligned = 0, fallback = 0;
    for (const auto& dims : shapes) {
        for (size_t start : {size_t{0}, size_t{8496}, size_t{32768} - dims[0]}) {
            run_case(dims[0], dims[1], start, 0, 0, prefix, tail);
            ++aligned;
        }
    }
    for (const auto& offsets : std::array<std::array<unsigned, 2>, 3>{{{1, 0}, {0, 1}, {1, 1}}}) {
        run_case(1, 8, 8496, offsets[0], offsets[1], prefix, tail);
        ++fallback;
    }
    alignas(4) std::array<uint16_t, 8193> in{}, out{};
    std::array<float, 128> angles{};
    auto* stream = static_cast<void*>(&in);
    const auto arm = deep_inverse_rope_h1_pack32_v1;
    const unsigned previous = launch_count;
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
    assert(launch_count == previous);
    injected_error = 73;
    assert(arm(out.data(), in.data(), angles.data(), 1, 8, stream) == 73 && last_packed_prefix);
    assert(arm(out.data() + 1, in.data(), angles.data(), 1, 8, stream) == 73 && !last_packed_prefix);
    assert(launch_count == previous + 2);
    std::cout << "{\"aligned_cases\":" << aligned << ",\"unaligned_fallback_cases\":" << fallback
              << ",\"prefix_values_exact\":" << prefix << ",\"rotated_values_exact\":" << tail
              << ",\"invalid_no_launch\":11,\"status_propagations\":2}\n";
}
