#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace infinilm::models::deepseek_v4::inverse_output {

enum class Arm { legacy, pack32, pack64 };

inline Arm requested_arm(const char* value) {
    const std::string_view name = value ? value : "";
    if (name.empty() || name == "legacy") {
        return Arm::legacy;
    }
    if (name == "h1-pack32") {
        return Arm::pack32;
    }
    if (name == "h1-pack64") {
        return Arm::pack64;
    }
    throw std::invalid_argument("Invalid DeepSeek inverse RoPE width for this candidate");
}

inline std::vector<float> make_table(const std::vector<std::int64_t>& positions,
                                     std::size_t max_position, double theta,
                                     double factor, double beta_fast, double beta_slow,
                                     std::size_t original) {
    for (auto position : positions) {
        if (position < 0 || std::size_t(position) >= max_position) {
            throw std::invalid_argument("DeepSeek rotary position out of bounds");
        }
    }
    constexpr double pi2 = 6.283185307179586476925286766559;
    double low = 0;
    double high = 0;
    if (original && factor != 1) {
        auto correction = [&](double rotations) {
            return 64 * std::log(double(original) / (rotations * pi2)) / (2 * std::log(theta));
        };
        low = std::clamp(std::floor(correction(beta_fast)), 0., 63.);
        high = std::clamp(std::ceil(correction(beta_slow)), 0., 63.);
        if (low == high) {
            high += .001;
        }
    }
    std::vector<float> table(positions.size() * 64);
    for (std::size_t token = 0; token < positions.size(); ++token) {
        for (std::size_t pair = 0; pair < 32; ++pair) {
            double frequency = 1 / std::pow(theta, double(2 * pair) / 64);
            if (original && factor != 1) {
                const double smooth = 1 - std::clamp((double(pair) - low) / (high - low), 0., 1.);
                frequency = (frequency / factor) * (1 - smooth) + frequency * smooth;
            }
            const double angle = double(positions[token]) * frequency;
            table[token * 64 + pair] = float(std::cos(angle));
            table[token * 64 + 32 + pair] = float(std::sin(angle));
        }
    }
    return table;
}

}  // namespace infinilm::models::deepseek_v4::inverse_output
