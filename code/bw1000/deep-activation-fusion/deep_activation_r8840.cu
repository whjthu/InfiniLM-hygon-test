#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <stdexcept>
namespace infinilm::models::deepseek_v4::activation_r8840 {
__global__ void fused(__nv_bfloat16*out,const __nv_bfloat16*in,size_t n){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 size_t j=(i/256)*512+i%256;
 float g=__bfloat162float(in[j]),u=__bfloat162float(in[j+256]);
 // Ordered comparisons propagate NaN and retain signed zero. ±10 are exact BF16.
 g=g>10.0f?10.0f:g;u=u< -10.0f? -10.0f:u;u=u>10.0f?10.0f:u;
 // Materialize original Clamp BF16 storage boundary before original Silu math.
 g=__bfloat162float(__float2bfloat16(g));u=__bfloat162float(__float2bfloat16(u));
 float sig=__frcp_rn(__fadd_rn(1.0f,__expf(-g)));
 out[i]=__float2bfloat16(__fmul_rn(__fmul_rn(g,sig),u));
}
void launch(void*out,const void*in,size_t n,void*stream){
 fused<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(out),static_cast<const __nv_bfloat16*>(in),n);
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
}
