#include "fresh_kv_route.hpp"

#include <cstdlib>
#include <cstring>

namespace glm_r474_fresh_kv_route {
namespace {
thread_local const void* active_owner = nullptr;
thread_local Mode current_mode = Mode::off;

Mode parse(const char* value) noexcept {
    if (value && std::strcmp(value, "64") == 0) {
        return Mode::tile64;
    }
    if (value && std::strcmp(value, "128") == 0) {
        return Mode::tile128;
    }
    return Mode::off;
}
}

Mode configured_mode() noexcept {
    static const Mode selected = parse(std::getenv("INFINILM_GLM474_FRESH_KV_PACK"));
    return selected;
}

Mode active_mode(const void* owner) noexcept {
    return owner && owner == active_owner ? current_mode : Mode::off;
}

ScopedMode::ScopedMode(const void* owner, Mode mode) noexcept
    : prior_owner_(active_owner), prior_mode_(current_mode) {
    active_owner = owner;
    current_mode = mode;
}

ScopedMode::~ScopedMode() {
    active_owner = prior_owner_;
    current_mode = prior_mode_;
}
}

extern "C" int glm_r474_fresh_kv_pack_t64(void*, void*, const void*, const void*, std::size_t, std::size_t, void*);
extern "C" int glm_r474_fresh_kv_pack_t128(void*, void*, const void*, const void*, std::size_t, std::size_t, void*);
