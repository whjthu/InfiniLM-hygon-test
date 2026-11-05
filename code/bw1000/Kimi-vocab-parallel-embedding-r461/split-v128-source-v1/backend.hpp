#pragma once
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "global_state/forward_context.hpp"
namespace kimi_split_v128 {
infinilm::runtime::Tensor forward(infinilm::runtime::Runtime&,const infinilm::runtime::Tensor&,const infinilm::runtime::Tensor&,const infinilm::runtime::Tensor&,const infinilm::runtime::Tensor&,const infinilm::global_state::AttentionMetadata&,float);
}
