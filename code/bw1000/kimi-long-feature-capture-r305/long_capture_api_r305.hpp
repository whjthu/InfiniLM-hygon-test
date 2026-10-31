#pragma once
#include <cstddef>
#include <cstdint>
namespace kimi_batch_live_r9122 {
// Hidden cross-TU helper. Only the public actual-Context wrapper may opt in.
__attribute__((visibility("hidden"))) int begin_long_validated_r305(void*,uint64_t,size_t,const int32_t*,const int32_t*,const int32_t*);
}
extern "C" int modern_kimi_batch_taps_begin_r9122(void*,uint64_t,int,size_t,const int32_t*,const int32_t*,const int32_t*);
