#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
namespace infini::ops::kimi_i192_r8669y {
__global__ void packRoutesI192R8669y(const int32_t*,int32_t*,int32_t*,size_t,size_t);
template<class T>__global__ void staticW2I192R8669y(T*,const T*,const int32_t*,const float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t,size_t);
}
extern "C" int kimi_A8_HIP_matrix_r8993(int,const void*,const void*,const void*,const void*,const void*,void*,void*);
namespace {
using B=__nv_bfloat16;
constexpr unsigned E=896,T=16,H=3584,I=192,N=384,R=256;
__global__ void situ(B*activation,const float*raw){
    const unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=R*I)return;
    const unsigned r=j/I,i=j%I;
    activation[j]=::mxfp4Store<B>(infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(raw[r*N+i],raw[r*N+I+i],size_t{2}));
}
}
// Same explicit pointer ABI as929; extra requires475648 bytes plus caller canary.
extern "C" int modern_kimi_A8_W13_r8993(int device,void*out,void*activation,const void*input,
 const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,
 void*counts,void*routes,void*extra,void*stream){
 int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=device||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!extra||stream)return int(cudaErrorInvalidValue);
 auto st=static_cast<cudaStream_t>(stream);auto*raw=static_cast<float*>(extra);
 auto code=cudaMemsetAsync(counts,0,E*sizeof(int),st);if(code!=cudaSuccess)return int(code);
 infini::ops::kimi_i192_r8669y::packRoutesI192R8669y<<<1,256,0,st>>>(static_cast<const int*>(ids),static_cast<int*>(counts),static_cast<int*>(routes),E,T);
 if(cudaGetLastError()!=cudaSuccess)return int(cudaErrorLaunchFailure);
 const int hip_status=kimi_A8_HIP_matrix_r8993(device,input,w13,s13,counts,routes,extra,stream);
 if(hip_status)return 10000+hip_status;
 situ<<<(R*I+255)/256,256,0,st>>>(static_cast<B*>(activation),raw);
 infini::ops::kimi_i192_r8669y::staticW2I192R8669y<B><<<H*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},E,H,I,size_t{0});
 return int(cudaGetLastError());
}
