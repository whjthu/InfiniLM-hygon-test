#include <cuda_runtime.h>
using B4 = unsigned short __attribute__((vector_size(8)));
using F4 = float __attribute__((vector_size(16)));
// Official NORMAL BF16 inputs/F32 accumulator. Both entry types are identical.
__global__ void official_bf16_mma_r8669zr(F4*o,const B4*a,const B4*b) {
  F4 c={};
#if __has_builtin(__builtin_hcu_mmac_f32_16x16x16_bf16)
  *o=__builtin_hcu_mmac_f32_16x16x16_bf16(*a,*b,c);
#else
  *o=c;
#endif
}
__global__ void proven_bf16_mma_r8669zr(F4*o,const B4*a,const B4*b) {
  F4 c={};*o=__builtin_amdgcn_mmac_f32_16x16x16bf16(*a,*b,c);
}
extern "C" int official_unsuffixed_supported_r8669zr(){
  return __has_builtin(__builtin_hcu_mmac_f32_16x16x16_bf16);
}
