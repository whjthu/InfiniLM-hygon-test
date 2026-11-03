#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace kimi_r440 {
inline bool physical_single_row(std::size_t ndim,std::size_t m,std::size_t k,std::size_t expected_k,std::int64_t rowpitch,std::int64_t inner){return ndim==2&&m==1&&k==expected_k&&rowpitch>0&&inner==1;}
// Address bound is Tensor::data + contiguous K BF16 elements. Rowpitch is unused
// for this one physical row; no assertion about private Runtime-pool offsets.
inline bool inside_owned_span(std::uintptr_t data,std::size_t k,std::uintptr_t base,std::size_t bytes){return k<=std::numeric_limits<std::size_t>::max()/2&&data>=base&&data-base<=bytes&&k*2<=bytes-(data-base);}
}
