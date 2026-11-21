#pragma once

#include <cstddef>
#include <stdexcept>

namespace glm_r474_fresh_kv_ar {
namespace detail {
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
inline thread_local const void* owner = nullptr;
}

inline bool active_for(const void* owner) noexcept {
    return owner && owner == detail::owner;
}

class Scope {
public:
    Scope(const void* owner, std::size_t tokens, std::size_t count, const std::size_t* requests) {
        if (!owner || detail::owner || !requests || tokens != 1 || count != 1
            || requests[0] != 0 || requests[1] == 0 || requests[2] != 1) {
            throw std::invalid_argument("fresh K/V requires one explicitly labeled decode continuation");
        }
        detail::owner = owner;
    }

    ~Scope() { detail::owner = nullptr; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
}
