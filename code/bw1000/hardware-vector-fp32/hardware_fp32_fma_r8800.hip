#include <hip/hip_runtime.h>
#include <math.h>
#include <stddef.h>
__global__ void register_fma8(const float* input,float* output,int loops){
 const size_t i=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*8;
 float a0=input[i],a1=input[i+1],a2=input[i+2],a3=input[i+3];
 float a4=input[i+4],a5=input[i+5],a6=input[i+6],a7=input[i+7];
 float c0=.125f,c1=.25f,c2=.375f,c3=.5f,c4=.625f,c5=.75f,c6=.875f,c7=1.f;
 const float b=1.f/64.f;
 #pragma unroll 1
 for(int k=0;k<loops;++k){
  c0=fmaf(a0,b,c0);c1=fmaf(a1,b,c1);c2=fmaf(a2,b,c2);c3=fmaf(a3,b,c3);
  c4=fmaf(a4,b,c4);c5=fmaf(a5,b,c5);c6=fmaf(a6,b,c6);c7=fmaf(a7,b,c7);
 }
 output[i]=c0;output[i+1]=c1;output[i+2]=c2;output[i+3]=c3;
 output[i+4]=c4;output[i+5]=c5;output[i+6]=c6;output[i+7]=c7;
}
extern "C" int hardware_fp32_fma_r8800(const void* input,void* output,int blocks,int loops,void* stream){
 if(!input||!output||blocks!=1280||loops!=65536)return 999;
 hipLaunchKernelGGL(register_fma8,dim3(blocks),dim3(256),0,static_cast<hipStream_t>(stream),static_cast<const float*>(input),static_cast<float*>(output),loops);
 return int(hipGetLastError());
}
