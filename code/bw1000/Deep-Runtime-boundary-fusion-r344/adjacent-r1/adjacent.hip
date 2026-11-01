#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cmath>
__device__ float loadbf(uint16_t x){return __uint_as_float(uint32_t(x)<<16);}
__device__ uint16_t bf(float x){uint32_t u=__float_as_uint(x);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
template<int B,bool F> __global__ __launch_bounds__(256) void norm(const uint16_t*x,const uint16_t*w,uint16_t*y,const float*table,int D,float eps){
 int row=blockIdx.x,l=threadIdx.x; x+=row*D;y+=row*D;
 __shared__ float red[B];float v=0;for(int d=l;d<D;d+=B){float z=loadbf(x[d]);v+=z*z;}red[l]=v;__syncthreads();
 for(int s=B/2;s;s/=2){if(l<s)red[l]+=red[l+s];__syncthreads();}float inv=rsqrtf(red[0]/D+eps);
 for(int d=l;d<D;d+=B){float z=loadbf(bf(loadbf(x[d])*inv*loadbf(w[d])));
  if(F&&d>=D-64){int k=(d-(D-64))/2;float c=table[row*64+k],s=table[row*64+32+k];int other=d^1;float q=loadbf(bf(loadbf(x[other])*inv*loadbf(w[other])));z=(d%2==0)?z*c-q*s:z*c+q*s;}
  y[d]=bf(z);
 }
}
__global__ void rope(uint16_t*y,const float*table,int D){int row=blockIdx.x,k=threadIdx.x;if(k>=32)return;y+=row*D+D-64;float a=loadbf(y[k]),b=loadbf(y[k+32]),c=table[row*64+k],s=table[row*64+32+k];y[k]=bf(a*c-b*s);y[k+32]=bf(b*c+a*s);}
extern "C" int run340(const void*x,const void*w,void*y,const void*table,int T,int D,int B,int fused,int rotate,float eps,void*stream){
 if(T<=0||(D!=128&&D!=512&&D!=4096)||fused<0||fused>1||rotate<0||rotate>1)return 900;
 auto st=(hipStream_t)stream;
#define GO(N) case N: if(fused&&rotate)hipLaunchKernelGGL((norm<N,true>),dim3(T),dim3(N),0,st,(const uint16_t*)x,(const uint16_t*)w,(uint16_t*)y,(const float*)table,D,eps);else hipLaunchKernelGGL((norm<N,false>),dim3(T),dim3(N),0,st,(const uint16_t*)x,(const uint16_t*)w,(uint16_t*)y,(const float*)table,D,eps);break;
 if(B!=256||!fused||!rotate)return 902;
 hipLaunchKernelGGL((norm<256,true>),dim3(T),dim3(256),0,st,(const uint16_t*)x,(const uint16_t*)w,(uint16_t*)y,(const float*)table,D,eps);

 return int(hipGetLastError());
}
