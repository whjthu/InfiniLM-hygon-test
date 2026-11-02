#include <hip/hip_runtime.h>
#include "hc_tail_contract.hpp"
// Two pointwise/custom launches are fused. The Runtime Mean, Add, Rsqrt,
// normalized Mul and F32Gemm remain their original operators and geometries.
__global__ __launch_bounds__(256) void cast_square(const uint16_t*bits,float*xf,float*square){
 for(unsigned i=blockIdx.x*256+threadIdx.x;i<16384;i+=gridDim.x*256){float x=hc372::unpack(bits[i]);xf[i]=x;square[i]=x*x;}
}
__global__ __launch_bounds__(256) void epilogue_sinkhorn(const float*xf,const float*mixes,const float*base,const float*scale,const float*constants,int iterations,uint16_t*collapsed,float*post,float*comb,unsigned*hit){
 __shared__ float pre[4],prob[16];unsigned lane=threadIdx.x;
 if(lane<4)hc372::row_weights(lane,mixes,base,scale,constants,pre,post,prob);
 __syncthreads();
 for(unsigned i=lane;i<4096;i+=256)collapsed[i]=hc372::collapse_at(xf,pre,i);
 if(lane==0){float a[16];for(int i=0;i<16;++i)a[i]=prob[i];hc372::sinkhorn(a,constants[1],iterations);for(int i=0;i<16;++i)comb[i]=a[i];atomicAdd(hit,1u);}
}
extern "C" int deep_hc_cast_square_r372(const void*bits,float*xf,float*square,void*stream){
 if(!bits||!xf||!square)return int(hipErrorInvalidValue);
 cast_square<<<64,256,0,reinterpret_cast<hipStream_t>(stream)>>>(static_cast<const uint16_t*>(bits),xf,square);return int(hipGetLastError());
}
extern "C" int deep_hc_tail_r372(const float*xf,const float*mixes,const float*base,const float*scale,const float*constants,int iterations,void*collapsed,float*post,float*comb,unsigned*hit,void*stream){
 if(!xf||!mixes||!base||!scale||!constants||!collapsed||!post||!comb||!hit||iterations<1||iterations>128)return int(hipErrorInvalidValue);
 epilogue_sinkhorn<<<1,256,0,reinterpret_cast<hipStream_t>(stream)>>>(xf,mixes,base,scale,constants,iterations,static_cast<uint16_t*>(collapsed),post,comb,hit);return int(hipGetLastError());
}
