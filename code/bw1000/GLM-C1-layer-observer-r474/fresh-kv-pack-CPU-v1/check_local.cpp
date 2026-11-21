#include "local_runtime.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

extern "C" int glm_r474_fresh_kv_pack_t64(void*, void*, const void*, const void*, size_t, size_t, void*);
extern "C" int glm_r474_fresh_kv_pack_t128(void*, void*, const void*, const void*, size_t, size_t, void*);

using Arm = decltype(&glm_r474_fresh_kv_pack_t64);
constexpr std::array<Arm, 2> arms{glm_r474_fresh_kv_pack_t64, glm_r474_fresh_kv_pack_t128};
constexpr size_t guard = 64;
constexpr uint16_t canary = 0xeab1;

void canaries(const std::vector<uint16_t>& v) {
    for (size_t i = 0; i < guard; ++i) {
        assert(v[i] == canary && v[v.size() - guard + i] == canary);
    }
}

int main() {
    int owner = 0;
    size_t key_words = 0, value_words = 0, shared_words = 0, passes = 0;
    for (size_t tokens : {size_t{1}, size_t{2}, size_t{17}, size_t{512}}) {
        for (unsigned seed : {17u, 19u, 73u}) {
            std::vector<uint16_t> x(tokens * 4 * 448 + 2 * guard, canary);
            std::vector<uint16_t> r(tokens * 64 + 2 * guard, canary);
            uint32_t state = seed;
            auto random_word = [&]() {
                state = state * 1664525u + 1013904223u;
                return static_cast<uint16_t>(state >> 16);
            };
            std::generate(x.begin() + guard, x.end() - guard, random_word);
            std::generate(r.begin() + guard, r.end() - guard, random_word);
            const auto frozen_x = x, frozen_r = r;
            std::vector<uint16_t> expected_k(tokens * 4 * 256), expected_v(tokens * 4 * 256);
            // Original Runtime decomposition. Non-rotary key copy, shared
            // per-head rotary copy, and value contiguous slice.
            for (size_t t = 0; t < tokens; ++t) {
                for (size_t h = 0; h < 4; ++h) {
                    const auto in = x.begin() + guard + (4 * t + h) * 448;
                    const auto k = expected_k.begin() + (4 * t + h) * 256;
                    std::copy_n(in, 192, k);
                    std::copy_n(r.begin() + guard + t * 64, 64, k + 192);
                    std::copy_n(in + 192, 256, expected_v.begin() + (4 * t + h) * 256);
                }
            }
            for (size_t variant = 0; variant < arms.size(); ++variant) {
                std::vector<uint16_t> k(expected_k.size() + 2 * guard, canary);
                std::vector<uint16_t> v(expected_v.size() + 2 * guard, canary);
                const unsigned threads = variant == 0 ? 64 : 128;
                const unsigned before = launch_count;
                assert(arms[variant](k.data() + guard, v.data() + guard, x.data() + guard, r.data() + guard,
                                     tokens, 4, &owner) == 0);
                assert(launch_count == before + 1 && last_grid == 4 * tokens * 128 / threads
                       && last_threads == threads && last_stream == &owner);
                assert(std::equal(expected_k.begin(), expected_k.end(), k.begin() + guard));
                assert(std::equal(expected_v.begin(), expected_v.end(), v.begin() + guard));
                canaries(k);
                canaries(v);
                assert(x == frozen_x && r == frozen_r);
                key_words += expected_k.size();
                value_words += expected_v.size();
                shared_words += tokens * 4 * 64;
                ++passes;
            }
        }
    }
    size_t rejected = 0;
    for (const auto arm : arms) {
        std::array<uint16_t, 2048> key{}, value{}, x{}, rotary{};
        auto* k = key.data();
        auto* v = value.data();
        auto* p = x.data();
        auto* r = rotary.data();
        const unsigned before = launch_count;
        assert(arm(nullptr, v, p, r, 1, 4, &owner) == 1);
        assert(arm(k, nullptr, p, r, 1, 4, &owner) == 1);
        assert(arm(k, v, nullptr, r, 1, 4, &owner) == 1);
        assert(arm(k, v, p, nullptr, 1, 4, &owner) == 1);
        assert(arm(k, v, p, r, 1, 4, nullptr) == 1);
        assert(arm(k, v, p, r, 0, 4, &owner) == 1);
        assert(arm(k, v, p, r, 513, 4, &owner) == 1);
        assert(arm(k, v, p, r, 1, 8, &owner) == 1);
        assert(arm(k, k, p, r, 1, 4, &owner) == 1);
        assert(arm(k, k + 1, p, r, 1, 4, &owner) == 1);
        assert(arm(k, v, k, r, 1, 4, &owner) == 1);
        assert(arm(k, v, p, v + 100, 1, 4, &owner) == 1);
        assert(arm(k, v, p, p + 42, 1, 4, &owner) == 1);
        assert(arm(reinterpret_cast<uint8_t*>(k) + 1, v, p, r, 1, 4, &owner) == 1);
        assert(launch_count == before);
        rejected += 14;
        injected_error = 71;
        assert(arm(k, v, p, r, 1, 4, &owner) == 71 && launch_count == before + 1);
        injected_error = 0;
    }
    std::cout << "{\"accepted\":" << passes << ",\"exact_key_words\":" << key_words
              << ",\"exact_value_words\":" << value_words << ",\"exact_rotary_key_words\":" << shared_words
              << ",\"rejected_no_launch\":" << rejected << ",\"launch_errors_propagated\":2}\n";
}
