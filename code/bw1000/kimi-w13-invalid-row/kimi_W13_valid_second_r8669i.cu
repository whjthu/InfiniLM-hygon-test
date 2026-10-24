#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdlib>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
#include "kimi_mxfp4_static_r326.cuh"
#include "kimi_tiled_w13_r1858.cuh"
namespace infini::ops::mxfp4_detail {
__global__ void packRoutes4531(const int32_t*,int32_t*,int32_t*,size_t,size_t);
template<int TT>__global__ void buildJobs4531(const int32_t*,int32_t*,int32_t*,int32_t*,size_t);
// Pair/group only routes to the SAME expert; keep the original product order,
// 256-lane reduction tree, activation and final W2 path for every route.
template<int TT,int Tile,bool SkipInvalidSecond=false>__global__ void groupedW13ValidR8669i(__nv_bfloat16*activated,const __nv_bfloat16*input,const int32_t*counts,const int32_t*routes,const int32_t*jobExpert,const int32_t*jobStart,const int32_t*jobs,const uint8_t*packed,const uint8_t*scales,size_t T){
 constexpr size_t H=3584,I=384,K=1792,S=112;extern __shared__ float scratch[];
 for(size_t work=blockIdx.x;work<size_t(jobs[0])*(I/Tile);work+=gridDim.x){
  size_t job=work/(I/Tile),first=work%(I/Tile)*Tile;int e=jobExpert[job],start=jobStart[job];int rows[TT];float sums[TT*2*Tile]={};
  #pragma unroll
  for(int t=0;t<TT;++t)rows[t]=(start+t<counts[e])?routes[size_t(e)*T*16+start+t]:-1;
  // CTA-uniform job geometry branch encloses WHOLE p-loop; avoids198
  // per-FMA if-conversion. Fullvalid path is exactoriginalTT2loop.
  if(!SkipInvalidSecond || rows[1]>=0){
  #pragma unroll 1
  for(size_t p=threadIdx.x;p<K;p+=256){
   float a[TT],b[TT];
   #pragma unroll
   for(int t=0;t<TT;++t){const auto*x=rows[t]>=0?input+(size_t(rows[t])/16)*H:input;a[t]=rows[t]>=0?mxfp4Load(x,2*p):0.f;b[t]=rows[t]>=0?mxfp4Load(x,2*p+1):0.f;}
   #pragma unroll
   for(int j=0;j<Tile;++j){size_t gate=size_t(e)*2*I+first+j,up=gate+I;float gl,gh,ul,uh;mxfp4DecodePair(packed[gate*K+p],scales[gate*S+p/16],gl,gh);mxfp4DecodePair(packed[up*K+p],scales[up*S+p/16],ul,uh);
    #pragma unroll
    for(int t=0;t<TT;++t){sums[(t*Tile+j)*2]+=a[t]*gl+b[t]*gh;sums[(t*Tile+j)*2+1]+=a[t]*ul+b[t]*uh;}
   }
  }
  }else{
  #pragma unroll 1
  for(size_t p=threadIdx.x;p<K;p+=256){
   float a[TT],b[TT];
   #pragma unroll
   for(int t=0;t<TT;++t){const auto*x=rows[t]>=0?input+(size_t(rows[t])/16)*H:input;a[t]=rows[t]>=0?mxfp4Load(x,2*p):0.f;b[t]=rows[t]>=0?mxfp4Load(x,2*p+1):0.f;}
   #pragma unroll
   for(int j=0;j<Tile;++j){size_t gate=size_t(e)*2*I+first+j,up=gate+I;float gl,gh,ul,uh;mxfp4DecodePair(packed[gate*K+p],scales[gate*S+p/16],gl,gh);mxfp4DecodePair(packed[up*K+p],scales[up*S+p/16],ul,uh);
    #pragma unroll
    for(int t=0;t<1;++t){sums[(t*Tile+j)*2]+=a[t]*gl+b[t]*gh;sums[(t*Tile+j)*2+1]+=a[t]*ul+b[t]*uh;}
   }
  }
  }
  reduceWaveR1044<TT*2*Tile,64>(sums,scratch);
  if(threadIdx.x==0){
   #pragma unroll
   for(int t=0;t<TT;++t)if(rows[t]>=0){
    #pragma unroll
    for(int j=0;j<Tile;++j)activated[size_t(rows[t])*I+first+j]=mxfp4Store<__nv_bfloat16>(fusedMoeMxfp4Activate(sums[(t*Tile+j)*2],sums[(t*Tile+j)*2+1],size_t{2}));
   }
  }
  __syncthreads();
 }
}
}
extern "C" int modern_kimi_grouped_r4620(int,int,void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,void*,void*,size_t,size_t,void*);

namespace infini::ops::mxfp4_detail {
extern template __global__ void staticW2R326U<__nv_bfloat16>(__nv_bfloat16*,const __nv_bfloat16*,const int32_t*,const float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t,size_t);
}
extern "C" int modern_kimi_valid_events_r8669i(int device,int variant,int mode,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*jobExpert,void*jobStart,void*jobs,void*partial,size_t E,size_t T,void*stream,void*begin_event,void*end_event) {
 using namespace infini::ops::mxfp4_detail;using B=__nv_bfloat16;int current=-1;
 if(cudaGetDevice(&current)!=cudaSuccess||current!=device||variant!=4||mode<0||mode>2||E!=896||T!=16||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!jobExpert||!jobStart||!jobs||!partial)return int(cudaErrorInvalidValue);
 if(mode==0)return modern_kimi_grouped_r4620(device,variant,out,activation,input,ids,weights,w13,s13,w2,s2,counts,routes,jobExpert,jobStart,jobs,partial,E,T,stream);
 auto st=static_cast<cudaStream_t>(stream);cudaError_t code;
 code=cudaMemsetAsync(counts,0,E*sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);code=cudaMemsetAsync(jobs,0,sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);
 packRoutes4531<<<(T*16+255)/256,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),E,T);
 buildJobs4531<2><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
 if(begin_event){code=cudaEventRecord(static_cast<cudaEvent_t>(begin_event),st);if(code!=cudaSuccess)return int(code);}
 // Mode0 is untouched originalprovider/defaultOFF; mode1 exactsourcecontrol,
 // mode2 only suppresses arithmetic for the absent secondrow. No row/store,
 // load/dequant, reduction, activation, scratch or W2 policy change.
 if(mode==2){groupedW13ValidR8669i<2,4,true><<<640,256,16384,st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);}else{groupedW13ValidR8669i<2,4,false><<<640,256,16384,st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);}
 if(end_event){code=cudaEventRecord(static_cast<cudaEvent_t>(end_event),st);if(code!=cudaSuccess)return int(code);}
 code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
 staticW2R326U<B><<<3584*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},E,size_t{3584},size_t{384},size_t{0});return int(cudaGetLastError());
}
