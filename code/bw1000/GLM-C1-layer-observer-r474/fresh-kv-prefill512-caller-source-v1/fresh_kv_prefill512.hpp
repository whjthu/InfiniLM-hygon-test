#pragma once

#include "fresh_kv_route.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace glm_r474_fresh_kv_prefill512 {

inline constexpr std::size_t tokens = 512;
inline constexpr std::size_t heads = 4;
inline constexpr std::size_t sections = 5;
enum Counter : std::size_t { attempted, packed, rotary_layout, tensor_layout, address_span };

class Scope;
namespace detail {
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
inline thread_local Scope* phase = nullptr;
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
inline thread_local const void* materializing_owner = nullptr;
}

class Scope {
public:
    Scope(const void* owner, std::size_t n, std::size_t count, const std::size_t* requests)
        : owner_(owner) {
        if (!owner || detail::phase || !requests || n != tokens || count != 1
            || requests[0] != 0 || requests[1] != 0 || requests[2] != tokens) {
            throw std::invalid_argument("fresh K/V trial requires the explicit initial C1 T512 prefill");
        }
        detail::phase = this;
    }

    ~Scope() { detail::phase = nullptr; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    bool owns(const void* owner) const noexcept { return owner_ == owner; }
    void increment(Counter counter) noexcept { ++counts_[counter]; }
    void publish(std::size_t* sink) const noexcept {
        for (std::size_t i = 0; i < counts_.size(); ++i) {
            sink[i] = counts_[i];
        }
    }

private:
    const void* owner_;
    std::array<std::size_t, sections> counts_{};
};

namespace detail {
class Materializing {
public:
    explicit Materializing(const void* owner) {
        if (materializing_owner) {
            throw std::logic_error("nested fresh T512 materialization");
        }
        materializing_owner = owner;
        phase->increment(attempted);
    }
    ~Materializing() { materializing_owner = nullptr; }
    Materializing(const Materializing&) = delete;
    Materializing& operator=(const Materializing&) = delete;
};
}

template<class Invoke>
decltype(auto) with_fresh(const void* owner, bool geometry, Invoke&& invoke) {
    if (!geometry || !detail::phase || !detail::phase->owns(owner)) {
        return std::forward<Invoke>(invoke)();
    }
    const detail::Materializing scope(owner);
    return std::forward<Invoke>(invoke)();
}

inline bool active_for(const void* owner) noexcept {
    return owner && owner == detail::materializing_owner;
}

inline void record(const void* owner, Counter counter) {
    if (!active_for(owner) || !detail::phase || !detail::phase->owns(owner)) {
        throw std::logic_error("fresh T512 route lacks its physical Runtime owner");
    }
    detail::phase->increment(counter);
}

namespace layout {
using TensorLayout = ::glm_r474_fresh_kv_route::TensorLayout;
inline constexpr TensorLayout projected{{tokens, heads, 448}, {1792, 448, 1}};
inline constexpr TensorLayout rotary{{tokens, 1, 64}, {64, 64, 1}};
inline constexpr TensorLayout output{{tokens, heads, 256}, {1024, 256, 1}};

constexpr std::size_t bytes(const TensorLayout& tensor) noexcept {
    return 2 * tensor.shape[0] * tensor.shape[1] * tensor.shape[2];
}

template<class Tensor, class Device, class DataType>
bool check(const Tensor& tensor, const Device& device, DataType bf16, const TensorLayout& expected) {
    return ::glm_r474_fresh_kv_route::matches(tensor, device, bf16, expected)
        && tensor->element_size() == 2 && tensor->nbytes() == bytes(expected);
}

struct Span {
    const void* address;
    std::size_t bytes;
};

inline bool disjoint(std::array<Span,4> spans) noexcept {
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto start = reinterpret_cast<std::uintptr_t>(spans[i].address);
        const auto length = spans[i].bytes;
        if (!start || (start % alignof(std::uint16_t)) || !length
            || length > std::numeric_limits<std::uintptr_t>::max() - start) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            const auto earlier = reinterpret_cast<std::uintptr_t>(spans[j].address);
            if ((start <= earlier && earlier - start < length)
                || (earlier < start && start - earlier < spans[j].bytes)) {
                return false;
            }
        }
    }
    return true;
}
}
}
