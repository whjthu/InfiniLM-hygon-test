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
using namespace infini::ops::mxfp4_detail;

namespace i8r113 {

#if defined(__CUDA_ARCH__)
// Exact gfx936 DU mapping, bound to DTK du_mma.hpp SHA19126534...:
// A rowmajor/B colmajor: lane&15 outputrow/column, (lane>>4)*8 K-byte block.
// Accumulator rowmajor: row=lane&15, col=(lane>>4)+4*i.
using I4r113=int __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void atom(const signed char*a,const signed char*b,int*out){
 unsigned l=threadIdx.x&63,row=l&15,k0=(l>>4)*8;union Bits{uint64_t u;long s;};Bits ax{0},bx{0};
 #pragma unroll
 for(unsigned i=0;i<8;++i){ax.u|=uint64_t(uint8_t(a[row*32+k0+i]))<<(8*i);bx.u|=uint64_t(uint8_t(b[row*32+k0+i]))<<(8*i);}
 I4r113 z={0,0,0,0};I4r113 c=__builtin_amdgcn_mmac_i32_16x16x32i8(ax.s,bx.s,z);
 #pragma unroll
 for(unsigned i=0;i<4;++i)out[row*16+(l>>4)+4*i]=c[i];
}
#endif

struct Binding{int device=-1;size_t T=0;void*stream=nullptr;unsigned char*base=nullptr;size_t bytes=0;};
thread_local Binding binding;
std::atomic<uint64_t> counts[8][3]{};
size_t bytes(size_t T){return T*16*(192+24+3584*4+4)+4;}
__device__ __forceinline__ signed char decode(uint8_t c){constexpr int q[8]={0,1,2,3,4,6,8,12};int x=q[c&7];return static_cast<signed char>((c&8)?-x:x);}
__global__ void constructed(int32_t*ids,size_t R){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<R)ids[i]=int32_t(i%896);}
__global__ void quantize(const __nv_bfloat16*a,signed char*q,float*scale,int*error,size_t R){
 size_t rg=blockIdx.x;if(rg>=R*6)return;unsigned k=threadIdx.x;__shared__ float vals[32];float x=mxfp4Load(a,rg*32+k);if(!isfinite(x)){atomicExch(error,1);x=0.f;}vals[k]=fabsf(x);__syncthreads();
 for(unsigned s=16;s;s/=2){if(k<s)vals[k]=fmaxf(vals[k],vals[k+s]);__syncthreads();}float sa=vals[0]>0?vals[0]/127.f:1.f;
 if(!isfinite(sa)||!(sa>0)){atomicExch(error,1);sa=1.f;}if(k==0)scale[rg]=sa;int v=int(nearbyintf(x/sa));q[rg*32+k]=static_cast<signed char>(max(-127,min(127,v)));
}
__global__ void matrix(float*partial,const signed char*q,const float*as,const int32_t*cnt,const int32_t*routes,const int32_t*expert,const int32_t*start,const int32_t*jobs,const uint8_t*w,const uint8_t*sc,size_t T){
#if defined(__CUDA_ARCH__)
 // v2 changes only ownership of C/scale accumulation and the shared transpose.
 // All64 lanes must execute MMAC, including paddedA rows: their B fragments
 // still contribute to other valid rows. The validrow predicate starts AFTER it.
 __shared__ __align__(16) signed char aa[512];__shared__ __align__(16) signed char bb[512];__shared__ __align__(16) float tmp[256];__shared__ int rr[16];
 const unsigned lane=threadIdx.x,mnative=lane&15,colbase=lane>>4,k0=colbase*8;size_t R=T*16;
 for(size_t work=blockIdx.x;work<size_t(jobs[0])*224;work+=gridDim.x){size_t job=work/224;unsigned n0=(work%224)*16;int e=expert[job],s=start[job];
  if(lane<16)rr[lane]=(s+int(lane)<cnt[e])?routes[size_t(e)*R+s+lane]:-1;__syncthreads();const int rid=rr[mnative];float sum[4]={0.f,0.f,0.f,0.f};
  #pragma unroll
  for(unsigned g=0;g<6;++g){
   for(unsigned p=lane;p<512;p+=64){unsigned m=p/32,k=p%32;int r=rr[m];aa[p]=r>=0?q[size_t(r)*192+g*32+k]:0;unsigned n=p/32;uint8_t byte=w[(size_t(e)*3584+n0+n)*96+g*16+k/2];bb[p]=decode(uint8_t((byte>>(4*(k&1)))&15));}__syncthreads();
   // Same bytepacking/builtin as qualified750/751 atom, now retain C in registers.
   union Bits{uint64_t u;long s;};Bits ax{0},bx{0};
   #pragma unroll
   for(unsigned i=0;i<8;++i){ax.u|=uint64_t(uint8_t(aa[mnative*32+k0+i]))<<(8*i);bx.u|=uint64_t(uint8_t(bb[mnative*32+k0+i]))<<(8*i);}
   I4r113 z={0,0,0,0};I4r113 c=__builtin_amdgcn_mmac_i32_16x16x32i8(ax.s,bx.s,z);
   if(rid>=0){const float av=as[size_t(rid)*6+g];
    #pragma unroll
    for(unsigned i=0;i<4;++i){unsigned n=colbase+4*i;float sw=ldexpf(.5f,int(sc[(size_t(e)*3584+n0+n)*6+g])-127);sum[i]+=float(c[i])*(av*sw);}
   }
   // Keep conservative A/B reuse barrier; no pergroup I32 shared roundtrip.
   __syncthreads();
  }
  if(rid>=0){
   #pragma unroll
   for(unsigned i=0;i<4;++i)tmp[mnative*16+colbase+4*i]=sum[i];
  }
  __syncthreads();
  // Exactly one final F32 rowmajor transpose, then the original coalesced egress.
  #pragma unroll
  for(unsigned i=0;i<4;++i){unsigned p=lane+64*i,m=p/16,n=p%16;int r=rr[m];if(r>=0)partial[size_t(r)*3584+n0+n]=tmp[p];}__syncthreads();
 }
#endif
}

__global__ void reduce(__nv_bfloat16*out,const float*partial,const int32_t*ids,const float*weight,size_t T){size_t p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=T*3584)return;size_t t=p/3584,n=p%3584;float v=0.f;for(unsigned k=0;k<16;++k){int e=ids[t*16+k];if(e>=0&&e<896)v+=weight[t*16+k]*partial[(t*16+k)*3584+n];}out[p]=mxfp4Store<__nv_bfloat16>(v);}
__global__ void layout(const signed char*a,const signed char*b,int*out){
#if defined(__CUDA_ARCH__)
atom(a,b,out);
#endif
}

}
extern "C" size_t modern_kimi_i8_bytes_r113(size_t T){return T>=8&&T<=64?i8r113::bytes(T):0;}
extern "C" int modern_kimi_i8_bind_r113(int device,size_t T,void*base,size_t bytes,void*stream){if(!base){i8r113::binding={};return 0;}if(device<0||device>=8||T<8||T>64||bytes!=i8r113::bytes(T)||reinterpret_cast<uintptr_t>(base)%16)return 1;i8r113::binding={device,T,stream,static_cast<unsigned char*>(base),bytes};return 0;}
extern "C" int modern_kimi_i8_counts_r113(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=i8r113::counts[d][i].load();return 0;}
extern "C" int modern_kimi_i8_layout_r113(const void*a,const void*b,void*out,void*stream){i8r113::layout<<<1,64,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const signed char*>(a),static_cast<const signed char*>(b),static_cast<int*>(out));return int(cudaGetLastError());}

// Host-only owner-thread events. Default disabled; no CUDA synchronization here.
namespace k114b {
struct State{int device=-1;cudaStream_t stream=nullptr;cudaEvent_t e[5]{};bool ready=false,on=false;unsigned marks=0;};
thread_local State p;
int mark(unsigned n,cudaStream_t stream){if(!p.on)return 0;if(!p.ready||p.stream!=stream||n>4)return int(cudaErrorInvalidValue);if(n==0)p.marks=0;auto c=cudaEventRecord(p.e[n],stream);if(c==cudaSuccess)p.marks|=1u<<n;return int(c);}
}
extern "C" int modern_kimi_k114b_init(int device,void*stream){auto&p=k114b::p;int current=-1;if(p.ready||cudaGetDevice(&current)!=cudaSuccess||device!=current)return int(cudaErrorInvalidValue);p.device=device;p.stream=static_cast<cudaStream_t>(stream);for(unsigned i=0;i<5;++i){auto c=cudaEventCreate(&p.e[i]);if(c!=cudaSuccess){for(unsigned j=0;j<i;++j)cudaEventDestroy(p.e[j]);p={};return int(c);}}p.ready=true;return 0;}
extern "C" int modern_kimi_k114b_enable(int on){if(on&&!k114b::p.ready)return int(cudaErrorInvalidValue);k114b::p.on=on!=0;return 0;}
extern "C" int modern_kimi_k114b_read(float*out,size_t n){auto&p=k114b::p;if(!out||n!=5||!p.on||p.marks!=31)return int(cudaErrorInvalidValue);for(unsigned i=0;i<4;++i){auto c=cudaEventElapsedTime(out+i,p.e[i],p.e[i+1]);if(c!=cudaSuccess)return int(c);}return int(cudaEventElapsedTime(out+4,p.e[0],p.e[4]));}
extern "C" int modern_kimi_k114b_close(){auto&p=k114b::p;p.on=false;int ret=0;if(p.ready)for(auto e:p.e){auto c=cudaEventDestroy(e);if(c!=cudaSuccess)ret=int(c);}p={};return ret;}
namespace{std::atomic<uint64_t>pair_counts_r9280[8][2]{};}
extern "C" int modern_kimi_W2_pair_counts_r9280(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=2)return 1;for(size_t i=0;i<2;++i)out[i]=pair_counts_r9280[device][i].load(std::memory_order_relaxed);return 0;}
extern "C" int modern_kimi_grouped_I192_r8669y(int device,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*jobExpert,void*jobStart,void*jobs,size_t E,size_t T,void*stream){
 using namespace infini::ops::kimi_i192_r8669y;using B=__nv_bfloat16;int current=-1;
 if(cudaGetDevice(&current)!=cudaSuccess||current!=device||E!=896||(T<8||T>64)||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!jobExpert||!jobStart||!jobs)return int(cudaErrorInvalidValue);
 const char*flag=std::getenv("INFINILM_KIMI_W2_INT8_R113");const bool i8=flag&&flag[0]=='1'&&flag[1]==0;
 const char*ctl=std::getenv("INFINILM_KIMI_W2_CONSTRUCTED_R113");const bool low=ctl&&ctl[0]=='1'&&ctl[1]==0;
 auto bind=i8r113::binding;if((i8||low)&&(bind.device!=device||bind.T!=T||bind.stream!=stream||!bind.base||bind.bytes!=i8r113::bytes(T)))return int(cudaErrorInvalidValue);
 const size_t R=T*16;auto*qA=reinterpret_cast<signed char*>(bind.base);auto*sA=reinterpret_cast<float*>(bind.base?bind.base+R*192:nullptr);auto*partial=reinterpret_cast<float*>(bind.base?bind.base+R*216:nullptr);auto*newids=reinterpret_cast<int32_t*>(bind.base?bind.base+R*(216+3584*4):nullptr);auto*error=reinterpret_cast<int*>(bind.base?bind.base+R*(216+3584*4+4):nullptr);
 auto st=static_cast<cudaStream_t>(stream);if(auto c=k114b::mark(0,st))return c;if(low){i8r113::constructed<<<(R+255)/256,256,0,st>>>(newids,R);ids=newids;}
 i8r113::counts[device][i8?1:0].fetch_add(1);if(low)i8r113::counts[device][2].fetch_add(1);
 auto code=cudaMemsetAsync(counts,0,E*sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);code=cudaMemsetAsync(jobs,0,sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);
 packRoutesI192R8669y<<<(T*16+255)/256,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),E,T);
 buildJobsI192R8669y<2><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
 groupedW13I192R8669y<2,4><<<640,256,2*2*4*256*sizeof(float),st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);
 code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
 if(auto c=k114b::mark(1,st))return c;
 if(i8){
  code=cudaMemsetAsync(error,0,sizeof(int),st);if(code!=cudaSuccess)return int(code);
  i8r113::quantize<<<R*6,32,0,st>>>(static_cast<const B*>(activation),qA,sA,error,R);
  code=cudaMemsetAsync(jobs,0,sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);
  buildJobsI192R8669y<16><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
  if(auto c=k114b::mark(2,st))return c;
  i8r113::matrix<<<640,64,0,st>>>(partial,qA,sA,static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T);
  if(auto c=k114b::mark(3,st))return c;
  i8r113::reduce<<<(T*3584+255)/256,256,0,st>>>(static_cast<B*>(out),partial,static_cast<const int32_t*>(ids),static_cast<const float*>(weights),T);if(auto c=k114b::mark(4,st))return c;return int(cudaGetLastError());
 }
 if(auto c=k114b::mark(2,st))return c;
 const char*toggle=std::getenv("INFINILM_KIMI_W2_PAIR_R9280");const bool enabled=toggle&&toggle[0]=='1'&&toggle[1]=='\0';
 pair_counts_r9280[device][enabled?1:0].fetch_add(1,std::memory_order_relaxed);
 if(enabled){pairW2I192R9280<<<1792*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,E);}
 else staticW2I192R8669y<<<3584*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},E,size_t{3584},size_t{192},size_t{0});if(auto c=k114b::mark(3,st))return c;if(auto c=k114b::mark(4,st))return c;return int(cudaGetLastError());
}
