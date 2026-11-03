#include <cuda_runtime.h>
#include "fp32_leaves_r446.h"
#include "fp32_bits_r446.hpp"
#include <cmath>
#include <cstdint>
#include <initializer_list>

namespace {
using namespace glm_value_r446;
__global__ __launch_bounds__(256) void cast_kernel(int direction,size_t n,const void*input,void*output,uint32_t*status){
 const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 if(direction==0){
  const auto u=bf16_to_f32_bits(static_cast<const uint16_t*>(input)[i]);
  if(!finite_bits(u))atomicOr(status,1u);
  static_cast<float*>(output)[i]=__uint_as_float(u);
 }else{
  const auto u=__float_as_uint(static_cast<const float*>(input)[i]);const auto b=f32_to_bf16_bits(u);
  if(!finite_bits(u)||!finite_bits(bf16_to_f32_bits(b)))atomicOr(status,2u);
  static_cast<uint16_t*>(output)[i]=b;
 }
}
// Fixed B256, one thread per output column, strict ascending-K F32 fmaf.
// No splitK, warp reduction, matrix instruction, hidden precision mode or grid.
// Different association from a vendor GEMM is explicit, never a lower dtype.
template<unsigned K,unsigned N>
__global__ __launch_bounds__(256) void gemv_kernel(const float*x,const float*w,float*y,uint32_t*status){
 const unsigned column=blockIdx.x*256+threadIdx.x;if(column>=N)return;
 float sum=0.f;
 for(unsigned k=0;k<K;++k)sum=fmaf(x[k],w[size_t(k)*N+column],sum);
 if(!finite_bits(__float_as_uint(sum)))atomicOr(status,4u);
 y[column]=sum;
}
int pointer_device(const void*p,int device,size_t alignment){
 if(!p||reinterpret_cast<uintptr_t>(p)%alignment)return 9441;
 cudaPointerAttributes a{};auto e=cudaPointerGetAttributes(&a,p);
 if(e!=cudaSuccess)return int(e);
 return a.type==cudaMemoryTypeDevice&&a.device==device?0:9442;
}
int guard(int device,void*stream,const void*a,const void*b,const void*c,const void*d){
 int actual=-1;auto e=cudaGetDevice(&actual);if(e!=cudaSuccess)return int(e);
 if(device<0||device>=8||actual!=device)return 9443;
 for(const void*p:{a,b,c,d}){const int r=pointer_device(p,device,2);if(r)return r;}
 cudaStreamCaptureStatus capture; e=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
 if(e!=cudaSuccess)return int(e);return capture==cudaStreamCaptureStatusNone?0:9444;
}
}
extern "C" int glm_value_cast_r446(int device,int direction,size_t n,const void*input,void*output,uint32_t*status,void*stream){
 if((direction!=0&&direction!=1)||(n!=16384&&n!=2048)||input==output)return 9440;
 int r=guard(device,stream,input,output,status,status);if(r)return r;
 if((r=pointer_device(status,device,4)))return r;
 if((r=pointer_device(direction==0?output:input,device,4)))return r;
 cast_kernel<<<unsigned((n+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(direction,n,input,output,status);
 return int(cudaGetLastError());
}
extern "C" int glm_value_gemv_r446(int device,int kind,const float*x,const float*w,float*y,uint32_t*status,void*stream){
 if((kind!=0&&kind!=1)||x==y||w==y)return 9440;
 int r=guard(device,stream,x,w,y,status);if(r)return r;
 for(const void*p:{static_cast<const void*>(x),static_cast<const void*>(w),static_cast<const void*>(y),static_cast<const void*>(status)})if((r=pointer_device(p,device,4)))return r;
 if(kind==0)gemv_kernel<2048,512><<<2,256,0,static_cast<cudaStream_t>(stream)>>>(x,w,y,status);
 else gemv_kernel<512,256><<<1,256,0,static_cast<cudaStream_t>(stream)>>>(x,w,y,status);
 return int(cudaGetLastError());
}
