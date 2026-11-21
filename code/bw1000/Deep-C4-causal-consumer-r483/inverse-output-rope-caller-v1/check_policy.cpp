#include "inverse_output_policy.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>

using namespace infinilm::models::deepseek_v4::inverse_output;

std::vector<float> reference(const std::vector<std::int64_t>&, double,
                             double, double, double, std::size_t);

int main() {
    const int owner = 1;
    assert(requested_arm(nullptr) == Arm::legacy);
    assert(requested_arm("") == Arm::legacy);
    assert(requested_arm("legacy") == Arm::legacy);
    assert(requested_arm("h1") == Arm::h1);
    assert(requested_arm("h2") == Arm::h2);
    bool invalid_arm = false;
    try {
        (void)requested_arm("unknown-arm");
    } catch (const std::invalid_argument&) {
        invalid_arm = true;
    }
    assert(invalid_arm);
    int decisions = 0;
    for (const auto arm : {Arm::legacy, Arm::h1, Arm::h2}) {
        assert(admitted_arm(arm, 1, 8, true, &owner) == arm);
        assert(admitted_arm(arm, 17, 8, true, &owner) == Arm::legacy);
        assert(admitted_arm(arm, 1, 64, true, &owner) == Arm::legacy);
        assert(admitted_arm(arm, 1, 8, false, &owner) == Arm::legacy);
        assert(admitted_arm(arm, 1, 8, true, nullptr) == Arm::legacy);
        decisions += 5;
    }
    int table_cases = 0;
    std::size_t words = 0;
    const std::array<std::vector<std::int64_t>, 3> positions = {{
        {0},
        {8496},
        {32767, 4095, 0, 1, 8191, 24, 4095},
    }};
    for (const auto& pos : positions) {
        for (const auto scaling : {1., 16.}) {
            for (const auto theta : {10000., 160000.}) {
                auto expected = reference(pos, theta, scaling, 32., 1., 65536);
                auto actual = make_table(pos, 32768, theta, scaling, 32., 1., 65536);
                assert(actual.size() == expected.size());
                assert(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
                words += actual.size();
                ++table_cases;
            }
        }
    }
    int rejected_positions = 0;
    for (auto value : {int64_t(-1), int64_t(32768), INT64_MAX}) {
        try {
            (void)make_table({value}, 32768, 160000., 16., 32., 1., 65536);
        } catch (const std::invalid_argument&) {
            ++rejected_positions;
        }
    }
    assert(rejected_positions == 3);
    std::cout << "{\"route_decisions\":" << decisions << ",\"valid_table_cases\":" << table_cases
              << ",\"bit_equal_F32_words\":" << words << ",\"invalid_positions\":"
              << rejected_positions << ",\"invalid_policy_rejected\":true}\n";
}
