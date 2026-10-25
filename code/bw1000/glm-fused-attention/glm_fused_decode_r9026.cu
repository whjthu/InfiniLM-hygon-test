#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cmath>
namespace {
struct Arguments {const __nv_bfloat16*q;const __nv_bfloat16*k[8];const __nv_bfloat16*v[8];unsigned p[8];__nv_bfloat16*out;};
__device__ float shuffle_down(float x,int n){
#if defined(__HIPCC__)
 return __shfl_down(x,n,64);
#else
 return __shfl_down_sync(0xffffffffu,x,n,64);
#endif
}
// Fixed current geometry only: exact per-row prefix, no padding in Softmax.
__global__ void fused_decode_8x8x256(Arguments a){
 const unsigned row=blockIdx.x,head=blockIdx.y,t=threadIdx.x,w=t/64,lane=t%64,P=a.p[row];
 __shared__ float score[640],reduce[256];
 __shared__ __nv_bfloat16 probability[640];
 const auto*q=a.q+(row*8+head)*256;
 for(unsigned p=w;p<P;p+=4){
  const auto*k=a.k[row]+(p*8+head)*256;float dot=0.f;
  #pragma unroll
  for(unsigned j=0;j<4;++j)dot=__fmaf_rn(float(q[lane+64*j]),float(k[lane+64*j]),dot);
  #pragma unroll
  for(unsigned delta=32;delta;delta/=2)dot=__fadd_rn(dot,shuffle_down(dot,delta));
  if(lane==0)score[p]=float(static_cast<__nv_bfloat16>(dot))*(1.f/16.f);
 }
 __syncthreads();
 float maximum=-INFINITY;for(unsigned p=t;p<P;p+=256)maximum=fmaxf(maximum,score[p]);
 reduce[t]=maximum;__syncthreads();
 for(unsigned n=128;n;n/=2){if(t<n)reduce[t]=fmaxf(reduce[t],reduce[t+n]);__syncthreads();}
 maximum=reduce[0];__syncthreads();float total=0.f;
 for(unsigned p=t;p<P;p+=256){float e=expf(score[p]-maximum);score[p]=e;total=__fadd_rn(total,e);}
 reduce[t]=total;__syncthreads();
 for(unsigned n=128;n;n/=2){if(t<n)reduce[t]=__fadd_rn(reduce[t],reduce[t+n]);__syncthreads();}
 total=reduce[0];
 for(unsigned p=t;p<P;p+=256)probability[p]=static_cast<__nv_bfloat16>(score[p]/total);
 __syncthreads();
 float value=0.f;
 for(unsigned p=0;p<P;++p)value=__fmaf_rn(float(probability[p]),float(a.v[row][(p*8+head)*256+t]),value);
 a.out[(row*8+head)*256+t]=static_cast<__nv_bfloat16>(value);
}
}
extern "C" int glm_fused_decode_r9026(void*out,const void*q,const void*const*keys,const void*const*values,const size_t*prefixes,size_t B,size_t H,size_t D,void*stream){
 if(!out||!q||!keys||!values||!prefixes||B!=8||H!=8||D!=256)return int(cudaErrorInvalidValue);
 Arguments a{};a.q=static_cast<const __nv_bfloat16*>(q);a.out=static_cast<__nv_bfloat16*>(out);
 for(unsigned i=0;i<8;++i){if(!keys[i]||!values[i]||!prefixes[i]||prefixes[i]>640)return int(cudaErrorInvalidValue);a.k[i]=static_cast<const __nv_bfloat16*>(keys[i]);a.v[i]=static_cast<const __nv_bfloat16*>(values[i]);a.p[i]=unsigned(prefixes[i]);}
 fused_decode_8x8x256<<<dim3(8,8),256,0,static_cast<cudaStream_t>(stream)>>>(a);
 return int(cudaGetLastError());
}
