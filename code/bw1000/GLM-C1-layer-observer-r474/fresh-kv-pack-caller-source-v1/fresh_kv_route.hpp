#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace glm_r474_fresh_kv_route {

enum class Mode { off, tile64, tile128 };

#if defined(__GNUC__) && !defined(_WIN32)
#define GLM_FRESH_HIDDEN __attribute__((visibility("hidden")))
#else
#define GLM_FRESH_HIDDEN
#endif

GLM_FRESH_HIDDEN Mode configured_mode() noexcept;
GLM_FRESH_HIDDEN Mode active_mode(const void* owner) noexcept;

class GLM_FRESH_HIDDEN ScopedMode {
public:
    ScopedMode(const void* owner, Mode mode) noexcept;
    ~ScopedMode();
    ScopedMode(const ScopedMode&) = delete;
    ScopedMode& operator=(const ScopedMode&) = delete;

private:
    const void* prior_owner_;
    Mode prior_mode_;
};
#undef GLM_FRESH_HIDDEN

template<class Invoke>
decltype(auto) with_fresh_call(const void* owner, bool single_continuation, Invoke&& invoke) {
    const Mode wanted = configured_mode();
    if (!single_continuation || wanted == Mode::off) {
        return std::forward<Invoke>(invoke)();
    }
    const ScopedMode scope(owner, wanted);
    return std::forward<Invoke>(invoke)();
}

struct TensorLayout {
    std::array<std::size_t, 3> shape;
    std::array<std::size_t, 3> stride;
};

inline constexpr TensorLayout projection{{1, 4, 448}, {1792, 448, 1}};
inline constexpr TensorLayout rotary_key{{1, 1, 64}, {64, 64, 1}};
inline constexpr TensorLayout key_or_value{{1, 4, 256}, {1024, 256, 1}};

template<class Tensor, class Device, class DataType>
bool matches(const Tensor& tensor, const Device& device, DataType bf16, const TensorLayout& layout) {
    if (!tensor || tensor->device() != device || tensor->dtype() != bf16
        || tensor->ndim() != 3 || !tensor->is_contiguous() || !tensor->data()) {
        return false;
    }
    for (std::size_t dim = 0; dim < layout.shape.size(); ++dim) {
        if (static_cast<std::size_t>(tensor->size(dim)) != layout.shape[dim]
            || static_cast<std::size_t>(tensor->stride(dim)) != layout.stride[dim]) {
            return false;
        }
    }
    return true;
}

}
