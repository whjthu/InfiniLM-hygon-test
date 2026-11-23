#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace infinilm::models::deepseek_v4::inverse_output {

enum class RouteIndex : std::size_t {
    legacy,
    fallback_geometry,
    fallback_no_stream,
    pack32,
    pack64,
    failed,
    count,
};

inline constexpr std::size_t route_count = static_cast<std::size_t>(RouteIndex::count);

inline bool route_counting_enabled() {
    static const bool enabled = [] {
        const char* flag = std::getenv("INFINILM_DEEP_INVERSE_ROPE_ROUTE_COUNTS");
        return flag && std::string_view(flag) == "1";
    }();
    return enabled;
}

inline std::array<std::atomic<std::uint64_t>, route_count>& route_counts() {
    struct Storage {
        std::array<std::atomic<std::uint64_t>, route_count> counts;

        Storage() {
            for (auto& value : counts) {
                value.store(0, std::memory_order_relaxed);
            }
        }
    };
    static Storage storage;
    return storage.counts;
}

class RouteSample {
public:
    RouteSample() : enabled_(route_counting_enabled()) {}
    RouteSample(const RouteSample&) = delete;
    RouteSample& operator=(const RouteSample&) = delete;

    ~RouteSample() {
        if (enabled_) {
            route_counts()[static_cast<std::size_t>(route_)].fetch_add(1, std::memory_order_relaxed);
        }
    }

    void complete(RouteIndex route) { route_ = route; }

private:
    bool enabled_;
    RouteIndex route_ = RouteIndex::failed;
};

inline int read_route_counts(std::uint64_t* values, std::size_t count) {
    if (!values || count != route_count) {
        return 1;
    }
    if (!route_counting_enabled()) {
        return 2;
    }
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = route_counts()[i].load(std::memory_order_relaxed);
    }
    return 0;
}

}  // namespace infinilm::models::deepseek_v4::inverse_output
