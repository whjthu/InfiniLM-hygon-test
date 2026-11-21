#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace deep_identity_swa_commit {

inline bool enabled(const char* value) noexcept {
    return value && value[0] == '1' && value[1] == '\0';
}

inline bool can_use_prepared_kv(bool opt_in, bool tensor_matches,
                                const std::vector<std::int64_t>& rows,
                                std::size_t slots, std::size_t tokens) noexcept {
    if (!opt_in || !tensor_matches || !tokens || rows.size() != tokens || slots != tokens) {
        return false;
    }
    for (std::size_t index = 0; index < tokens; ++index) {
        if (rows[index] < 0 || static_cast<std::uint64_t>(rows[index]) != index) {
            return false;
        }
    }
    return true;
}

}
