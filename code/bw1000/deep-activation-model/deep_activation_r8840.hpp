#pragma once
#include "runtime/runtime.hpp"
namespace infinilm::models::deepseek_v4::activation_r8840 {
extern thread_local bool enabled;
extern thread_local unsigned long long launches;
void launch(void*,const void*,size_t,void*);
void apply(runtime::Runtime&,const runtime::Tensor&,runtime::Tensor);
}
