// Private I192 specialization only: current TT2/Tile4 route grouping,
// original FP4 decode/FP32 products/reduction/Situ/BF16 output, no new grid.
#include <cuda_bf16.h>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
#include "kimi_tiled_w13_r1858.cuh"
namespace infini::ops::mxfp4_detail {
template<int TT>__global__ void buildJobs4531(const int32_t*,int32_t*,int32_t*,int32_t*,size_t);
}
namespace infini::ops::kimi_i192_r8669y {
using namespace infini::ops::mxfp4_detail;
__global__ void packRoutesI192R8669y(const int32_t*ids,int32_t*counts,int32_t*routes,size_t E,size_t T){
 size_t r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=T*16)return;int e=ids[r];if(e<0||size_t(e)>=E)return;int slot=atomicAdd(counts+e,1);if(slot<int(T*16))routes[size_t(e)*T*16+slot]=int32_t(r);
}
template<bool Direct>__device__ __forceinline__ void decodeW13R161(uint8_t packed,uint8_t scale,float&low,float&high){
 if constexpr(!Direct){mxfp4DecodePair(packed,scale,low,high);return;}
 if(scale<2||scale>252){mxfp4DecodePair(packed,scale,low,high);return;}
 const unsigned c0=packed&15,c1=packed>>4,m0=c0&7,m1=c1&7;
 const unsigned b0=m0==0?((c0&8)<<28):((c0&8)<<28)|((unsigned(scale)+(m0>>1)-1)<<23)|(m0==1?0u:(m0&1)<<22);
 const unsigned b1=m1==0?((c1&8)<<28):((c1&8)<<28)|((unsigned(scale)+(m1>>1)-1)<<23)|(m1==1?0u:(m1&1)<<22);
 low=__uint_as_float(b0);high=__uint_as_float(b1);
}
template<int TT,int Tile,bool Direct=false>__global__ void groupedW13I192R8669y(__nv_bfloat16*activated,const __nv_bfloat16*input,const int32_t*counts,const int32_t*routes,const int32_t*jobExpert,const int32_t*jobStart,const int32_t*jobs,const uint8_t*packed,const uint8_t*scales,size_t T){
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
   for(int j=0;j<Tile;++j){size_t gate=size_t(e)*2*I+first+j,up=gate+I;float gl,gh,ul,uh;decodeW13R161<Direct>(packed[gate*K+p],scales[gate*S+p/16],gl,gh);decodeW13R161<Direct>(packed[up*K+p],scales[up*S+p/16],ul,uh);
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

// One CTA, same token, two adjacent output channels. Independent original FP32
// lane reductions and ascending16route final accumulation for each channel.
template <typename T>
__global__ void pairW2I192R9280(T*output,const T*activated,const int32_t*ids,const float*weights,
 const uint8_t*packed,const uint8_t*scales,size_t tokens,size_t experts){
 constexpr size_t topk=16,hidden=3584,intermediate=192;
 const size_t token=blockIdx.x/(hidden/2),channel=2*(blockIdx.x%(hidden/2));
 if(token>=tokens)return;
 const size_t lane=threadIdx.x%32,group=threadIdx.x/32;
 __shared__ float sums[2][16];
 #pragma unroll 2
 for(size_t tile=0;tile<topk;tile+=8){
  const size_t route_id=tile+group;float sum0=0.f,sum1=0.f;
  if(route_id<topk){
   const int32_t expert=ids[token*topk+route_id];
   if(expert>=0 && static_cast<size_t>(expert)<experts){
    const size_t row0=static_cast<size_t>(expert)*hidden+channel,row1=row0+1;
    #pragma unroll 6
    for(size_t p=lane;p<intermediate/2;p+=32){
     const float x0=mxfp4Load(activated,(token*topk+route_id)*intermediate+2*p);
     const float x1=mxfp4Load(activated,(token*topk+route_id)*intermediate+2*p+1);
     float low0,high0,low1,high1;
     mxfp4DecodePair(packed[row0*(intermediate/2)+p],scales[row0*(intermediate/32)+p/16],low0,high0);
     mxfp4DecodePair(packed[row1*(intermediate/2)+p],scales[row1*(intermediate/32)+p/16],low1,high1);
     sum0+=x0*low0+x1*high0;sum1+=x0*low1+x1*high1;
    }
   }
  }
  for(int shift=16;shift>0;shift/=2){
#if defined(__HIPCC__)
   sum0+=__shfl_down(sum0,shift,32);sum1+=__shfl_down(sum1,shift,32);
#else
   sum0+=__shfl_down_sync(0xffffffffu,sum0,shift,32);sum1+=__shfl_down_sync(0xffffffffu,sum1,shift,32);
#endif
  }
  if(lane==0 && route_id<topk){sums[0][route_id]=sum0;sums[1][route_id]=sum1;}
 }
 __syncthreads();
 if(threadIdx.x==0){
  float value0=0.f,value1=0.f;
  for(size_t r=0;r<topk;++r){
   const int32_t expert=ids[token*topk+r];
   if(expert>=0 && static_cast<size_t>(expert)<experts){
    value0+=weights[token*topk+r]*sums[0][r];value1+=weights[token*topk+r]*sums[1][r];
   }
  }
  output[token*hidden+channel]=mxfp4Store<T>(value0);
  output[token*hidden+channel+1]=mxfp4Store<T>(value1);
 }
}
}
namespace{std::atomic<uint64_t>pair_counts_r9280[8][2]{};}
extern "C" int modern_kimi_W2_pair_counts_r9280(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=2)return 1;for(size_t i=0;i<2;++i)out[i]=pair_counts_r9280[device][i].load(std::memory_order_relaxed);return 0;}
namespace{std::atomic<uint64_t> w13_decode_counts_r161[8][2]{};}
extern "C" int modern_kimi_W13_decode_counts_r161(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=2)return 1;for(size_t i=0;i<2;++i)out[i]=w13_decode_counts_r161[device][i].load(std::memory_order_relaxed);return 0;}
extern "C" int modern_kimi_moe_phase_mark_r161(unsigned,void*);
extern "C" int modern_kimi_grouped_I192_r8669y(int device,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*jobExpert,void*jobStart,void*jobs,size_t E,size_t T,void*stream){
 using namespace infini::ops::kimi_i192_r8669y;using B=__nv_bfloat16;int current=-1;
 if(cudaGetDevice(&current)!=cudaSuccess||current!=device||E!=896||(T<8||T>64)||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!jobExpert||!jobStart||!jobs)return int(cudaErrorInvalidValue);
 if(modern_kimi_moe_phase_mark_r161(3,stream))return int(cudaErrorInvalidValue);
 auto st=static_cast<cudaStream_t>(stream);auto code=cudaMemsetAsync(counts,0,E*sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);code=cudaMemsetAsync(jobs,0,sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);
 if(modern_kimi_moe_phase_mark_r161(4,stream))return int(cudaErrorInvalidValue);
 packRoutesI192R8669y<<<(T*16+255)/256,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),E,T);
 if(modern_kimi_moe_phase_mark_r161(5,stream))return int(cudaErrorInvalidValue);
 ::infini::ops::mxfp4_detail::buildJobs4531<2><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
 if(modern_kimi_moe_phase_mark_r161(6,stream))return int(cudaErrorInvalidValue);
 const char*partial=std::getenv("INFINILM_KIMI_W13_DIRECTDECODE_R161");const bool use_direct=partial&&partial[0]=='1'&&partial[1]=='\0';
 w13_decode_counts_r161[device][use_direct?1:0].fetch_add(1,std::memory_order_relaxed);
 if(use_direct){groupedW13I192R8669y<2,4,true><<<640,256,2*2*4*256*sizeof(float),st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);}
 else{groupedW13I192R8669y<2,4><<<640,256,2*2*4*256*sizeof(float),st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);}
 if(modern_kimi_moe_phase_mark_r161(7,stream))return int(cudaErrorInvalidValue);
 code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
 const char*toggle=std::getenv("INFINILM_KIMI_W2_PAIR_R9280");const bool enabled=toggle&&toggle[0]=='1'&&toggle[1]=='\0';
 pair_counts_r9280[device][enabled?1:0].fetch_add(1,std::memory_order_relaxed);
 if(enabled){pairW2I192R9280<<<1792*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,E);}
 else staticW2I192R8669y<<<3584*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},E,size_t{3584},size_t{192},size_t{0});if(modern_kimi_moe_phase_mark_r161(8,stream))return int(cudaErrorInvalidValue);return int(cudaGetLastError());
}
