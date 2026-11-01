#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
// E4M3FN RN-even, saturation exactly at finite 448; NaN is rejected by caller.
__device__ uint8_t fp8(float x){
 unsigned sign=__float_as_uint(x)>>31;float a=fminf(fabsf(x),448.f);unsigned v;
 if(a<0.015625f)v=(unsigned)rintf(a*512.f);
 else{int e=int((__float_as_uint(a)>>23)&255)-127;int m=int(rintf(ldexpf(a,3-e)));if(m==16){++e;m=8;}v=unsigned((e+7)*8+m-8);}
 return uint8_t((sign<<7)|v);
}
template<int L,int G> __global__ void fused(const __nv_bfloat16*g,const __nv_bfloat16*u,uint8_t*out,float*sc,int blocks){
 __shared__ float sm[L*G];int lane=threadIdx.x%L,group=threadIdx.x/L,b=blockIdx.x*G+group;
 float v[128/L],mx=0;
 #pragma unroll
 for(int j=0;j<128/L;++j){int i=b*128+lane+j*L;float z=0;if(b<blocks){float x=__bfloat162float(g[i]);__nv_bfloat16 s=__float2bfloat16(x/(1.f+expf(-x)));z=__bfloat162float(__float2bfloat16(__bfloat162float(s)*__bfloat162float(u[i])));}v[j]=z;mx=fmaxf(mx,fabsf(z));}
 sm[threadIdx.x]=mx;__syncthreads();
 #pragma unroll
 for(int d=L/2;d;d/=2){if(lane<d)sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+d]);__syncthreads();}
 float scale=fmaxf(sm[group*L],1e-10f)/448.f;if(b<blocks){if(lane==0)sc[b]=scale;
 #pragma unroll
 for(int j=0;j<128/L;++j)out[b*128+lane+j*L]=fp8(v[j]/scale);}
}
extern "C" int glm_fused341(const void*g,const void*u,void*out,void*sc,int rows,int variant,void*stream){
 if(!g||!u||!out||!sc||rows<=0||rows>32768||variant<0||variant>=16)return -1;
 int blocks=rows*2;
 #define CASE(I,L,G) case I:fused<L,G><<<(blocks+G-1)/G,L*G,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,(uint8_t*)out,(float*)sc,blocks);break;
 switch(variant){CASE(0,16,1) CASE(1,16,2) CASE(2,16,4) CASE(3,16,8) CASE(4,32,1) CASE(5,32,2) CASE(6,32,4) CASE(7,32,8) CASE(8,64,1) CASE(9,64,2) CASE(10,64,4) CASE(11,64,8) CASE(12,128,1) CASE(13,128,2) CASE(14,128,4) CASE(15,128,8)}
 return int(cudaGetLastError());
}
