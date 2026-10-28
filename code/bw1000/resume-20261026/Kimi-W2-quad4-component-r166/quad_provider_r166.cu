// r166: one additional output-reuse kernel. Original503 device kernels reused.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
namespace infini::ops::kimi_i192_r8669y {
__global__ void packRoutesI192R8669y(const int32_t*,int32_t*,int32_t*,size_t,size_t);
template<int TT,int Tile,bool Partial>__global__ void groupedW13I192R8669y(__nv_bfloat16*,const __nv_bfloat16*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const uint8_t*,const uint8_t*,size_t);
template<class T>__global__ void pairW2I192R9280(T*,const T*,const int32_t*,const float*,const uint8_t*,const uint8_t*,size_t,size_t);
}
namespace infini::ops::mxfp4_detail {template<int TT>__global__ void buildJobs4531(const int32_t*,int32_t*,int32_t*,int32_t*,size_t);}
namespace kimi_quad_r166 {
using namespace infini::ops::mxfp4_detail;
__global__ void quadW2(__nv_bfloat16*output,const __nv_bfloat16*activated,const int32_t*ids,const float*weights,const uint8_t*packed,const uint8_t*scales,size_t tokens,size_t experts){
 constexpr size_t H=3584,I=192,K=96,S=6,C=4;
 const size_t token=blockIdx.x/(H/C),channel=C*(blockIdx.x%(H/C));if(token>=tokens)return;
 const unsigned lane=threadIdx.x%32,group=threadIdx.x/32;__shared__ float sums[C][16];
 #pragma unroll 2
 for(unsigned tile=0;tile<16;tile+=8){
  const unsigned route=tile+group;float accum[C]={};
  if(route<16){const int e=ids[token*16+route];if(e>=0&&size_t(e)<experts){
   #pragma unroll 6
   for(size_t p=lane;p<K;p+=32){
    const float x0=mxfp4Load(activated,(token*16+route)*I+2*p),x1=mxfp4Load(activated,(token*16+route)*I+2*p+1);
    #pragma unroll
    for(unsigned c=0;c<C;++c){const size_t row=size_t(e)*H+channel+c;float low,high;mxfp4DecodePair(packed[row*K+p],scales[row*S+p/16],low,high);accum[c]+=x0*low+x1*high;}
   }
  }}
  for(unsigned shift=16;shift;shift/=2){
   #pragma unroll
   for(unsigned c=0;c<C;++c){
#if defined(__HIPCC__)
    accum[c]+=__shfl_down(accum[c],shift,32);
#else
    accum[c]+=__shfl_down_sync(0xffffffffu,accum[c],shift,32);
#endif
   }
  }
  if(lane==0&&route<16){
   #pragma unroll
   for(unsigned c=0;c<C;++c)sums[c][route]=accum[c];
  }
 }
 __syncthreads();
 if(threadIdx.x==0){float values[C]={};for(unsigned r=0;r<16;++r){const int e=ids[token*16+r];if(e>=0&&size_t(e)<experts){const float w=weights[token*16+r];
   #pragma unroll
   for(unsigned c=0;c<C;++c)values[c]+=w*sums[c][r];
  }}
  #pragma unroll
  for(unsigned c=0;c<C;++c)output[token*H+channel+c]=mxfp4Store<__nv_bfloat16>(values[c]);
 }
}
struct EventPair{cudaEvent_t begin{},end{};bool created=false,armed=false,recorded=false;int device=-1;void*stream=nullptr;};
thread_local EventPair ep;std::atomic<uint64_t>hits[8][2]{};
int record(bool before,int device,void*stream){if(!ep.armed)return 0;if(ep.device!=device||ep.stream!=stream)return int(cudaErrorInvalidValue);auto s=cudaEventRecord(before?ep.begin:ep.end,static_cast<cudaStream_t>(stream));if(!before){ep.armed=false;ep.recorded=s==cudaSuccess;}return int(s);}
}
extern "C" int modern_kimi_W2_quad_counts_r166(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=2)return 1;for(unsigned i=0;i<2;++i)out[i]=kimi_quad_r166::hits[device][i].load(std::memory_order_relaxed);return 0;}
extern "C" int modern_kimi_W2_event_arm_r166(int device,void*stream){using namespace kimi_quad_r166;int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=device||ep.armed||ep.recorded)return 1;if(!ep.created){if(cudaEventCreate(&ep.begin)!=cudaSuccess||cudaEventCreate(&ep.end)!=cudaSuccess)return 1;ep.created=true;}ep.device=device;ep.stream=stream;ep.armed=true;return 0;}
extern "C" int modern_kimi_W2_event_read_r166(float*out){using namespace kimi_quad_r166;if(!out||ep.armed||!ep.recorded||cudaEventQuery(ep.end)!=cudaSuccess)return 1;auto code=cudaEventElapsedTime(out,ep.begin,ep.end);ep.recorded=false;return int(code);}
extern "C" int modern_kimi_W2_event_close_r166(){using namespace kimi_quad_r166;if(ep.armed||ep.recorded)return 1;if(ep.created){if(cudaEventDestroy(ep.begin)!=cudaSuccess||cudaEventDestroy(ep.end)!=cudaSuccess)return 1;}ep=EventPair{};return 0;}
extern "C" int modern_kimi_moe_phase_mark_r155(unsigned,void*);
extern "C" int modern_kimi_grouped_I192_r8669y(int device,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*jobExpert,void*jobStart,void*jobs,size_t E,size_t T,void*stream){
 using namespace infini::ops::kimi_i192_r8669y;using B=__nv_bfloat16;int current=-1;
 if(cudaGetDevice(&current)!=cudaSuccess||current!=device||E!=896||(T<8||T>64)||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!jobExpert||!jobStart||!jobs)return int(cudaErrorInvalidValue);
 if(modern_kimi_moe_phase_mark_r155(3,stream))return int(cudaErrorInvalidValue);auto st=static_cast<cudaStream_t>(stream);
 auto code=cudaMemsetAsync(counts,0,E*4,st);if(code!=cudaSuccess)return int(code);code=cudaMemsetAsync(jobs,0,4,st);if(code!=cudaSuccess)return int(code);
 if(modern_kimi_moe_phase_mark_r155(4,stream))return int(cudaErrorInvalidValue);
 packRoutesI192R8669y<<<(T*16+255)/256,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),E,T);
 if(modern_kimi_moe_phase_mark_r155(5,stream))return int(cudaErrorInvalidValue);
 ::infini::ops::mxfp4_detail::buildJobs4531<2><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
 if(modern_kimi_moe_phase_mark_r155(6,stream))return int(cudaErrorInvalidValue);
 groupedW13I192R8669y<2,4,false><<<640,256,16384,st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);
 if(modern_kimi_moe_phase_mark_r155(7,stream))return int(cudaErrorInvalidValue);code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
 const char*toggle=std::getenv("INFINILM_KIMI_W2_QUAD_R166");const bool quad=toggle&&toggle[0]=='1'&&toggle[1]=='\0';kimi_quad_r166::hits[device][quad?1:0].fetch_add(1,std::memory_order_relaxed);
 if(kimi_quad_r166::record(true,device,stream))return int(cudaErrorInvalidValue);
 if(quad){kimi_quad_r166::quadW2<<<896*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,E);}
 else{pairW2I192R9280<B><<<1792*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,E);}
 if(kimi_quad_r166::record(false,device,stream))return int(cudaErrorInvalidValue);
 if(modern_kimi_moe_phase_mark_r155(8,stream))return int(cudaErrorInvalidValue);return int(cudaGetLastError());
}
