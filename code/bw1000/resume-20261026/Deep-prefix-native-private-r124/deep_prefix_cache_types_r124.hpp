#pragma once
#include "runtime/runtime.hpp"
#include "deep_prefix_transaction_r124.hpp"
namespace infinilm::models::deepseek_v4::prefix_r124 {
struct PackedSlot {runtime::Tensor raw;size_t slot,block;std::array<uint8_t,256>valid{};};
}
