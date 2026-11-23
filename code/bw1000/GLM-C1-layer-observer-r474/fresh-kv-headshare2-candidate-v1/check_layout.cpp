#define GLM_HEADSHARE_HOST_CHECK
#include "fresh_kv_headshare.cu"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t rows = tokens_per_call * heads;
    std::vector<uint16_t> projected(rows * projected_width);
    std::vector<uint16_t> rotary(tokens_per_call * rotary_width);
    for (std::size_t i = 0; i < projected.size(); ++i) {
        projected[i] = uint16_t(971 * i + 3);
    }
    for (std::size_t i = 0; i < rotary.size(); ++i) {
        rotary[i] = uint16_t(113 * i + 187);
    }
    std::vector<uint16_t> key1(rows * output_width + 2, 0x1234), value1(key1);
    std::vector<uint16_t> key2(key1), value2(key1);
    for (std::size_t block = 0; block < rows; ++block) {
        for (unsigned lane = 0; lane < threads; ++lane) {
            copy_lane<1>(key1.data() + 1, value1.data() + 1, projected.data(), rotary.data(), block, lane);
            if (block < rows / 2) {
                copy_lane<2>(key2.data() + 1, value2.data() + 1, projected.data(), rotary.data(), block, lane);
            }
        }
    }
    assert(key1 == key2 && value1 == value2);
    assert(key2.front() == 0x1234 && key2.back() == 0x1234);
    assert(value2.front() == 0x1234 && value2.back() == 0x1234);
    for (std::size_t row = 0; row < rows; ++row) {
        auto src = projected.begin() + row * projected_width;
        auto tail = rotary.begin() + (row / heads) * rotary_width;
        auto dest = key2.begin() + row * output_width + 1;
        assert(std::equal(src, src + key_prefix, dest));
        assert(std::equal(tail, tail + rotary_width, dest + key_prefix));
        assert(std::equal(src + key_prefix, src + projected_width, value2.begin() + row * output_width + 1));
    }
    void* stream = reinterpret_cast<void*>(uintptr_t(8));
    assert(valid(key2.data() + 1, value2.data() + 1, projected.data(), rotary.data(), 512, 4, stream));
    assert(!valid(key2.data(), value2.data(), projected.data(), rotary.data(), 1, 4, stream));
    assert(!valid(key2.data(), value2.data(), projected.data(), rotary.data(), 512, 8, stream));
    assert(!valid(key2.data(), value2.data(), projected.data(), rotary.data(), 512, 4, nullptr));
    assert(!valid(nullptr, value2.data(), projected.data(), rotary.data(), 512, 4, stream));
    assert(!valid(key2.data(), key2.data() + 1, projected.data(), rotary.data(), 512, 4, stream));
    assert(!valid(key2.data(), value2.data(), projected.data(), projected.data() + 32, 512, 4, stream));
    auto* odd = reinterpret_cast<char*>(key2.data()) + 1;
    assert(!valid(odd, value2.data(), projected.data(), rotary.data(), 512, 4, stream));
    const auto* overflow = reinterpret_cast<void*>(UINTPTR_MAX - 1);
    assert(!valid(key2.data(), value2.data(), projected.data(), overflow, 512, 4, stream));
    std::cout << "1048576 BF16 elements per variant matched independent references; 8 bad bindings rejected\n";
}
