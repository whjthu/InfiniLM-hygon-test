#include <cuda_runtime.h>
#include <cstdint>
#include "contract.hpp"
namespace {
__device__ __forceinline__ float bf16(uint16_t x){return __uint_as_float(uint32_t(x)<<16);}
__device__ __forceinline__ uint16_t output_bf16(float x){unsigned u=__float_as_uint(x);return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);}
// One fixed 256-thread block per output row. No matrix-core, FP4 or shape grid.
__global__ void gemv(const uint16_t* __restrict__ x,const uint16_t* __restrict__ w,uint16_t* __restrict__ y,size_t n,size_t k){
 const size_t row=blockIdx.x;const unsigned t=threadIdx.x;__shared__ float sum[256];float a=0.0f;
 if(row>=n)return;
 for(size_t j=t;j<k;j+=256)a=fmaf(bf16(w[row*k+j]),bf16(x[j]),a);
 sum[t]=a;__syncthreads();
 for(unsigned s=128;s;s>>=1){if(t<s)sum[t]+=sum[t+s];__syncthreads();}
 if(t==0)y[row]=output_bf16(sum[0]);
}
}
extern "C" int kimi_lowrank_gemv_r432(int projection,void*y,const void*x,const void*w,size_t n,size_t k,size_t stride,void*stream){
 try{kimi_r432::validate(projection,1,n,k,stride);}catch(...){return 7001;}
 if(!x||!w||!y||x==y||w==y)return 7002;
 gemv<<<dim3(unsigned(n)),dim3(256),0,static_cast<cudaStream_t>(stream)>>>(static_cast<const uint16_t*>(x),static_cast<const uint16_t*>(w),static_cast<uint16_t*>(y),n,k);
 return int(cudaGetLastError());
}
