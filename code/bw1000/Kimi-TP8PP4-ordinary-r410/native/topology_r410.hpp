#pragma once
#include <cstddef>
#include <stdexcept>
namespace kimi_tp8_pp4_r410 {
// This is a new ordinary-only admission. Original TP16/PP2 factories stay scoped.
inline bool topology(bool optin, std::size_t tp, std::size_t pp) {
    return optin ? tp == 8 && pp == 4 : tp == 16 && pp == 2;
}
inline void cache_scope(bool optin, std::size_t tp, std::size_t pp) {
    if (!topology(optin, tp, pp))
        throw std::invalid_argument("r410 explicit long cache topology");
}
inline void call_scope(int kind, std::size_t count, int row, int past,
                       std::size_t tokens, int flags) {
    if ((kind != 0 && kind != 1) || count != 1 || row != 1 || past < 0 ||
        !tokens || tokens > (kind == 0 ? 512u : 1u) ||
        (kind == 1 && past == 0) || tokens > 32768u ||
        std::size_t(past) > 32768u - tokens || flags != 3)
        throw std::invalid_argument("r410 C1 prompt512/decode1/context32768");
}
}
