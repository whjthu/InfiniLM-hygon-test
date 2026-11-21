#include <array>
#include <cassert>
#include <iostream>

using namespace kimi_output_residual_scores_v1;

float tree(std::array<float, lanes> terms) {
    for (unsigned stride = lanes / 2; stride; stride /= 2) {
        for (unsigned lane = 0; lane < stride; ++lane) {
            terms[lane] = add(terms[lane], terms[lane + stride]);
        }
    }
    return terms[0];
}

void check_numerical() {
    std::array<std::uint16_t, width> source{}, norm{}, projection{};
    std::array<RetainedValues, lanes> held{};
    std::array<float, lanes> squares{}, dots{};
    for (unsigned col = 0; col < width; ++col) {
        norm[col] = encode(float(int((col * 61) % 237) - 118) / 64.0f);
        projection[col] = encode(float(int((col * 127) % 331) - 165) / 256.0f);
    }
    unsigned count = 0;
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned col = 0; col < width; ++col) {
            source[col] = encode(float(int((col * 109 + row * 137) % 1021) - 510) / 64.0f);
        }
        source[row % width] = 0x8000;
        source[(row + 1) % width] = 0x0001;
        for (unsigned lane = 0; lane < lanes; ++lane) {
            squares[lane] = retain_and_square(source.data(), lane, held[lane]);
            assert(__float_as_uint(squares[lane]) == __float_as_uint(square_lane(source.data(), lane)));
            for (unsigned sample = 0; sample < values_per_lane; ++sample) {
                assert(held[lane].words[sample] == source[lane + sample * lanes]);
            }
        }
        const float inv_rms = 1.0f / sqrtf(add(mul(tree(squares), 1.0f / float(width)), 1.0e-6f));
        std::array<float, lanes> original_dots{};
        for (unsigned lane = 0; lane < lanes; ++lane) {
            dots[lane] = dot_retained(held[lane], norm.data(), projection.data(), inv_rms, lane);
            original_dots[lane] = dot_lane(source.data(), norm.data(), projection.data(), inv_rms, lane);
            assert(__float_as_uint(dots[lane]) == __float_as_uint(original_dots[lane]));
        }
        assert(encode(tree(dots)) == encode(tree(original_dots)) && std::isfinite(tree(dots)));
        ++count;
    }
    std::cout << "{\"exact_bf16_scores\":" << count << ",\"exact_mapped_bf16_words\":" << count * width
              << ",\"square_and_dot_lane_pairs\":" << count * lanes << "}\n";
}
