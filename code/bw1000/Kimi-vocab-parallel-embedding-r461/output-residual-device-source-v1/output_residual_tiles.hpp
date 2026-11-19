#pragma once

#include <cstddef>

namespace kimi_output_residual_tile64 {

inline constexpr std::size_t full_tokens = 512;
inline constexpr std::size_t residuals = 9;
inline constexpr std::size_t hidden = 7168;
inline constexpr std::size_t tile_tokens = 64;
inline constexpr std::size_t tile_count = full_tokens / tile_tokens;
inline constexpr std::size_t original_fp32_route_min_rows = 512;

static_assert(full_tokens % tile_tokens == 0);
static_assert(tile_tokens * residuals >= original_fp32_route_min_rows);

inline bool selected(
    const char *setting, bool exact_owner,
    std::size_t rows, std::size_t tokens, std::size_t width,
    std::size_t states) noexcept {
    return setting && setting[0] == '1' && setting[1] == '\0'
        && exact_owner && rows == 1 && tokens == full_tokens
        && width == hidden && states == residuals;
}

struct Tile {
    std::size_t begin;
    std::size_t count;
};

inline constexpr Tile tile(std::size_t index) noexcept {
    return {index * tile_tokens, tile_tokens};
}

} // namespace kimi_output_residual_tile64
