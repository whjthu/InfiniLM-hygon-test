#pragma once
#include <cstddef>
#include <cstdint>
namespace kimi401 {
constexpr int role(std::size_t rows, std::size_t columns) {
    return ((rows == 768 && columns == 3584) || (rows == 384 && columns == 3584)) ? 0 :
           ((rows == 3584 && columns == 384) || (rows == 3584 && columns == 192)) ? 2 : -1;
}
constexpr std::size_t packed_per(std::size_t rows, std::size_t columns) { return rows * (columns / 2); }
static_assert(packed_per(384,3584) == packed_per(3584,384));
static_assert(role(384,3584) == 0 && role(3584,384) == 2);
static_assert(packed_per(384,3584) == 688128 && packed_per(3584,192) == 344064);
static_assert(896ULL * 1376256 < UINT32_MAX && 64ULL * 1376256 < UINT32_MAX/2);
}
