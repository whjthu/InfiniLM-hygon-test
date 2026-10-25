// Private I192 specialization only: current TT2/Tile4 route grouping,
// original FP4 decode/FP32 products/reduction/Situ/BF16 output, no new grid.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
#include "kimi_tiled_w13_r1858.cuh"
namespace infini::ops::kimi_i192_r8669y {
using namespace infini::ops::mxfp4_detail;
__global__ void packRoutesI192R8669y(const int32_t*ids,int32_t*counts,int32_t*routes,size_t E,size_t T){
 size_t r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=T*16)return;int e=ids[r];if(e<0||size_t(e)>=E)return;int slot=atomicAdd(counts+e,1);if(slot<int(T*16))routes[size_t(e)*T*16+slot]=int32_t(r);
}
template<int TT>__global__ void buildJobsI192R8669y(const int32_t*counts,int32_t*expert,int32_t*start,int32_t*jobs,size_t E){
 size_t e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=E)return;int n=(counts[e]+TT-1)/TT;if(!n)return;int first=atomicAdd(jobs,n);for(int i=0;i<n;++i){expert[first+i]=int32_t(e);start[first+i]=i*TT;}
}
template<int TT,int Tile>__global__ void groupedW13I192R8669y(__nv_bfloat16*activated,const __nv_bfloat16*input,const int32_t*counts,const int32_t*routes,const int32_t*jobExpert,const int32_t*jobStart,const int32_t*jobs,const uint8_t*packed,const uint8_t*scales,size_t T){
 constexpr size_t H=3584,I=192,K=1792,S=112;extern __shared__ float scratch[];
 for(size_t work=blockIdx.x;work<size_t(jobs[0])*(I/Tile);work+=gridDim.x){
  size_t job=work/(I/Tile),first=work%(I/Tile)*Tile;int e=jobExpert[job],start=jobStart[job];int rows[TT];float sums[TT*2*Tile]={};
  #pragma unroll
  for(int t=0;t<TT;++t)rows[t]=(start+t<counts[e])?routes[size_t(e)*T*16+start+t]:-1;
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
template <typename T>
__global__ void staticW2I192R8669y(
    T* output,const T* activated,const int32_t* ids,const float* weights,
    const uint8_t* packed,const uint8_t* scales,size_t tokens,size_t unused_topk,
    size_t experts,size_t unused_hidden,size_t unused_intermediate,size_t offset) {
    constexpr size_t topk=16,hidden=3584,intermediate=192;
    const size_t index=offset+blockIdx.x, token=index/hidden, channel=index%hidden;
    if(token>=tokens)return;
    const size_t lane=threadIdx.x%32, group=threadIdx.x/32;
    __shared__ float sums[16];
    #pragma unroll 2
    for(size_t tile=0;tile<topk;tile+=8) {
        const size_t route_id=tile+group;
        float sum=0.0f;
        if(route_id<topk) {
            const int32_t expert=ids[token*topk+route_id];
            if(expert>=0 && static_cast<size_t>(expert)<experts) {
                const size_t row=static_cast<size_t>(expert)*hidden+channel;
                #pragma unroll 6
                for(size_t p=lane;p<intermediate/2;p+=32) {
                    float low,high;
                    mxfp4DecodePair(packed[row*(intermediate/2)+p],scales[row*(intermediate/32)+p/16],low,high);
                    sum+=mxfp4Load(activated,(token*topk+route_id)*intermediate+2*p)*low
                        +mxfp4Load(activated,(token*topk+route_id)*intermediate+2*p+1)*high;
                }
            }
        }
        for(int shift=16;shift>0;shift/=2) {
#if defined(__HIPCC__)
            sum+=__shfl_down(sum,shift,32);
#else
            sum+=__shfl_down_sync(0xffffffffu,sum,shift,32);
#endif
        }
        if(lane==0 && route_id<topk)sums[route_id]=sum;
    }
    __syncthreads();
    if(threadIdx.x==0) {
        float value=0.0f;
        for(size_t r=0;r<topk;++r) {
            const int32_t expert=ids[token*topk+r];
            if(expert>=0 && static_cast<size_t>(expert)<experts)
                value+=weights[token*topk+r]*sums[r];
        }
        output[index]=mxfp4Store<T>(value);
    }
}
}
extern "C" int modern_kimi_grouped_I192_r8669y(int device,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*jobExpert,void*jobStart,void*jobs,size_t E,size_t T,void*stream){
 using namespace infini::ops::kimi_i192_r8669y;using B=__nv_bfloat16;int current=-1;
 if(cudaGetDevice(&current)!=cudaSuccess||current!=device||E!=896||(T<8||T>64)||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!jobExpert||!jobStart||!jobs)return int(cudaErrorInvalidValue);
 auto st=static_cast<cudaStream_t>(stream);auto code=cudaMemsetAsync(counts,0,E*sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);code=cudaMemsetAsync(jobs,0,sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);
 packRoutesI192R8669y<<<(T*16+255)/256,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),E,T);
 buildJobsI192R8669y<2><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
 groupedW13I192R8669y<2,4><<<640,256,2*2*4*256*sizeof(float),st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);
 code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
 staticW2I192R8669y<<<3584*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},E,size_t{3584},size_t{192},size_t{0});return int(cudaGetLastError());
}
