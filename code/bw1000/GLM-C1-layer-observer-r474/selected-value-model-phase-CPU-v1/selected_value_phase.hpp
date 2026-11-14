#pragma once

#include <cstddef>
#include <stdexcept>

namespace glm_selected_value_layer0 {
namespace detail {
inline thread_local const void* decode_owner = nullptr;
}

// One scope per explicitly labeled candidate forward on the Runtime owner thread.
class ForwardScope {
public:
    ForwardScope(const void* owner, int use_packed, std::size_t tokens,
                 std::size_t requests, const std::size_t* metadata) {
        if (!owner || (use_packed != 0 && use_packed != 1) || detail::decode_owner)
            throw std::invalid_argument("invalid selected-value forward scope");
        if (use_packed == 1) {
            if (!metadata || tokens != 1 || requests != 1 ||
                metadata[0] != 0 || metadata[1] == 0 || metadata[2] != 1)
                throw std::invalid_argument("packed selected-value requires a single decode continuation");
            detail::decode_owner = owner;
        }
    }
    ~ForwardScope() { detail::decode_owner = nullptr; }
    ForwardScope(const ForwardScope&) = delete;
    ForwardScope& operator=(const ForwardScope&) = delete;
};

inline bool eligible(const void* owner, std::size_t count, std::size_t rows) noexcept {
    return owner && owner == detail::decode_owner && count == 1 && rows >= 65 && rows <= 2048;
}
}
