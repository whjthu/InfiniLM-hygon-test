#include "depth_detail_plan.hpp"
#include <cassert>

using glm_r474_depth_detail::select;

int main() {
    const std::array<std::array<std::size_t, 2>, 3> expected{{
        {{18, 19}}, {{38, 40}}, {{66, 69}},
    }};
    for (unsigned ordinal = 0; ordinal < 6; ++ordinal) {
        assert(select(false, ordinal) == nullptr);
        const auto* pair = select(true, ordinal);
        if (ordinal < 1 || ordinal > expected.size()) {
            assert(pair == nullptr);
            continue;
        }
        assert(pair && pair->ordinal == ordinal);
        for (std::size_t layer = 0; layer < 78; ++layer) {
            assert(pair->accepts(layer, true) == (layer == expected[ordinal - 1][0]));
            assert(pair->accepts(layer, false) == (layer == expected[ordinal - 1][1]));
        }
        assert(!pair->accepts(78, true) && !pair->accepts(78, false));
    }
}
