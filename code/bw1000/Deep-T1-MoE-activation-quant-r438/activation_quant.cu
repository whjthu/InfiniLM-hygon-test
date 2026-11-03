#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
// Same BF16 Clamp/Silu store boundary as SDK d54a84..., followed by exact
// QuantI8-0 scale/max tree and half-away integer rule from SDK32d7a4....
// No GPU F64, block grid, altered I8Gemm, events or extra stream/fence.
__global__ __launch_bounds__(256) void fused_r438(const __nv_bfloat16*gu,int8_t*packed,float*scales,__nv_bfloat16*diagnostic){
 const size_t row=blockIdx.x;const unsigned c=threadIdx.x;__shared__ float maximum[256];
 float g=__bfloat162float(gu[row*512+c]),u=__bfloat162float(gu[row*512+c+256]);
 g=g>10.f?10.f:g;u=u< -10.f? -10.f:u;u=u>10.f?10.f:u;
 g=__bfloat162float(__float2bfloat16(g));u=__bfloat162float(__float2bfloat16(u));
 const float sig=__frcp_rn(__fadd_rn(1.f,__expf(-g)));
 const auto rounded=__float2bfloat16(__fmul_rn(__fmul_rn(g,sig),u));
 const float activation=__bfloat162float(rounded);
 if(diagnostic)diagnostic[row*256+c]=rounded;
 maximum[c]=fmaxf(0.f,fabsf(activation));__syncthreads();
 for(unsigned step=128;step;step/=2){if(c<step)maximum[c]=fmaxf(maximum[c],maximum[c+step]);__syncthreads();}
 const float s=fmaxf(maximum[0]/127.f,1e-8f);if(c==0)scales[row]=s;
 const float inverse=1.f/s;float v=activation*inverse;
 int q=v>=0?int(floorf(fabsf(v)+.5f)):-int(floorf(fabsf(v)+.5f));q=q>127?127:(q< -127?-127:q);packed[row*256+c]=int8_t(q);
}
extern "C" int deep_activation_quant_launch_r438(const void*gu,void*packed,void*scales,void*diagnostic,size_t rows,void*stream){
 if(!gu||!packed||!scales||(rows!=1&&rows!=6))return -1;
 fused_r438<<<unsigned(rows),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(gu),static_cast<int8_t*>(packed),static_cast<float*>(scales),static_cast<__nv_bfloat16*>(diagnostic));
 return int(cudaGetLastError());
}
