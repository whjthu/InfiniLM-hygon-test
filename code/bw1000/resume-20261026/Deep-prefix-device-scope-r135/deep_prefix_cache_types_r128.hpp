#pragma once
#include "runtime/runtime.hpp"
#include "deep_prefix_transaction_r128.hpp"
namespace infinilm::models::deepseek_v4::prefix_r128 {
struct PackedSlot {runtime::Tensor raw;size_t slot,block;std::array<uint8_t,256>valid{};};
}
