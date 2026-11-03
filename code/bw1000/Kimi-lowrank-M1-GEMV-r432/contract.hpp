#pragma once
#include <cstddef>
#include <stdexcept>
namespace kimi_r432 {
struct Geometry {size_t n,k;};
inline Geometry geometry(int projection){if(projection==0)return{3584,7168};if(projection==1)return{7168,3584};throw std::invalid_argument("432 projection must be down0 or up1");}
inline void validate(int projection,int mode,size_t n,size_t k,size_t row_stride){auto g=geometry(projection);if(mode<0||mode>1||n!=g.n||k!=g.k||row_stride!=k)throw std::invalid_argument("432 only current lowrank M1 BF16 row-major exact geometry");}
}
