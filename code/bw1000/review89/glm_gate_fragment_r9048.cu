#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
// Each original lane receives exactly its original K indices; no Wtree change.
__device__ __forceinline__ uint32_t shuffle_word8528(uint32_t word,int owner,int width){
#if defined(__HIPCC__)
 return __shfl(word,owner,width);
#else
 return __shfl_sync(0xffffffffu,word,owner,width);
#endif
}
__device__ __forceinline__ uint8_t take_gate8528(uint32_t word,unsigned lane,int j){
 uint32_t peer=shuffle_word8528(word,int(lane/2)+j*32,64);
 return uint8_t(peer>>(8*(lane%2)));
}
__device__ __forceinline__ uint8_t take_down8528(uint32_t word,unsigned lane,int j){
 uint32_t peer=shuffle_word8528(word,int(lane/4)+j*8,32);
 return uint8_t(peer>>(8*(lane%4)));
}

__device__ float decode(uint8_t v) {
  unsigned e=(v>>3)&15,m=v&7;
  float x=e?(e==15&&m==7?nanf(""):ldexpf(1.f+float(m)/8,int(e)-7)):ldexpf(float(m),-9);
  return v&128?-x:x;
}
// SOURCE-ONLY candidate. Not compiled or launched. Default OFF caller required.
// MMAC inline-assembly idiom reference: Copyright (c) 2024, The vLLM team.
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
// Remaining fixed-route kernel/indexing is project-specific implementation.
// gfx936 MMAC operand/output mapping cross-checked against author primary source:
// sinpeyw/qwen3.5-vllm-dcu-optimization/csrc/rocm/attention.cu, lines5082..5107.
// Uses the CURRENT original decode(uint8_t) unchanged. No persistent/full-BF16 W.
using glm_mmac_b16x4_r9048 = uint16_t __attribute__((vector_size(8)));
using glm_mmac_f32x4_r9048 = float __attribute__((vector_size(16)));
__device__ __forceinline__ uint16_t glm_mmac_bf16_r9048(uint8_t x) {
 union { __nv_bfloat16 b; uint16_t u; } v;
 v.b=static_cast<__nv_bfloat16>(decode(x)); return v.u;
}
__device__ __forceinline__ glm_mmac_f32x4_r9048 glm_mmac_r9048(
 glm_mmac_b16x4_r9048 a,glm_mmac_b16x4_r9048 b,glm_mmac_f32x4_r9048 c) {
 asm volatile("s_nop 4\n\tv_mmac_f32_16x16x16_bf16 %0, %1, %2, %0\n\ts_nop 4\n\t"
              : "+v"(c) : "v"(a),"v"(b)); return c;
}
__global__ void gate_up_register_mmac_r9048(__nv_bfloat16*gate,__nv_bfloat16*up,
 const uint8_t*x,const float*xs,const int64_t*ids,const uint8_t*wg,const float*sg,
 const uint8_t*wu,const float*su,size_t routes,size_t offset) {
 // Four wave64s/CTA, each produces sixteen output rows for ONE actual route.
 // MMAC M=16 with only row0 populated. No padding fake requests into service B.
 unsigned lane=threadIdx.x%64,m=lane%16,group=lane/16;
 size_t nbase=size_t(blockIdx.x)*64+(threadIdx.x/64)*16;
 size_t route=blockIdx.y;if(route>=routes||nbase>=256)return;
 size_t token=route/8,e=size_t(ids[route]);
 glm_mmac_f32x4_r9048 g={0,0,0,0},u={0,0,0,0};
 for(size_t block=0;block<48;++block) {
  glm_mmac_f32x4_r9048 gd={0,0,0,0},ud={0,0,0,0};
  for(size_t sub=0;sub<8;++sub) {
   glm_mmac_b16x4_r9048 a={0,0,0,0},b,c;
   #pragma unroll
   for(unsigned j=0;j<4;++j) {
    size_t k=block*128+sub*16+group*4+j;
    if(m==0)a[j]=glm_mmac_bf16_r9048(x[token*6144+k]);
    size_t wi=(e*256+nbase+m)*6144+k;
    b[j]=glm_mmac_bf16_r9048(wg[wi]);c[j]=glm_mmac_bf16_r9048(wu[wi]);
   }
   gd=glm_mmac_r9048(a,b,gd);ud=glm_mmac_r9048(a,c,ud);
  }
  if(m==0) {
   #pragma unroll
   for(unsigned j=0;j<4;++j) {
    size_t n=nbase+group+j*4;
    size_t si=e*16*48+((offset+n)/128)*48+block;
    // Original K128 scale placement and block0..47 accumulation order.
    // INTRA-K128 reduction changes to MMAC; final Model IDs/EOS remain required.
    g[j]+=gd[j]*xs[token*48+block]*sg[si];
    u[j]+=ud[j]*xs[token*48+block]*su[si];
   }
  }
 }
 if(m==0) {
  #pragma unroll
  for(unsigned j=0;j<4;++j) {
   size_t n=nbase+group+j*4;
   gate[route*256+n]=static_cast<__nv_bfloat16>(g[j]);
   up[route*256+n]=static_cast<__nv_bfloat16>(u[j]);
  }
 }
}
// Planned launch: dim3(4,routes), block256, same stream; down8528 unchanged.

__global__ void validate(const int64_t* ids,size_t count,size_t experts,int* status){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count&&(ids[i]<0||size_t(ids[i])>=experts))atomicExch(status,1);
}
template<int W> __device__ float reduce(float* v){
  constexpr int V=128/W;
  #pragma unroll
  for(int stride=V/2;stride;stride/=2){
    #pragma unroll
    for(int j=0;j<V;++j)if(j<stride)v[j]=__fadd_rn(v[j],v[j+stride]);
  }
  float x=v[0];
  #pragma unroll
  for(int shift=W/2;shift;shift/=2){
#if defined(__HIPCC__)
    x=__fadd_rn(x,__shfl_down(x,shift,W));
#else
    x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,shift,W));
#endif
  }
  return x;
}
__global__ void gate_up(__nv_bfloat16* gate,__nv_bfloat16* up,const uint8_t* x,const float* xs,
 const int64_t* ids,const uint8_t* wg,const float* sg,const uint8_t* wu,const float* su,
 size_t routes,size_t offset){
  constexpr int W=64,V=2;
  size_t row=size_t(blockIdx.x)*4+threadIdx.x/W,route=blockIdx.y;
  if(row>=256||route>=routes)return;
  unsigned lane=threadIdx.x%W;size_t token=route/8,e=size_t(ids[route]);
  float g=0.f,u=0.f;
  for(size_t b=0;b<48;++b){
    float gv[V],uv[V];
    const uint32_t xv=*reinterpret_cast<const uint16_t*>(x+token*6144+b*128+2*lane);
    const uint32_t gw=*reinterpret_cast<const uint16_t*>(wg+(e*256+row)*6144+b*128+2*lane);
    const uint32_t uw=*reinterpret_cast<const uint16_t*>(wu+(e*256+row)*6144+b*128+2*lane);
    #pragma unroll
    for(int j=0;j<V;++j){
      size_t k=b*128+lane+size_t(j)*W;
      float a=decode(take_gate8528(xv,lane,j));
      gv[j]=__fadd_rn(a*decode(take_gate8528(gw,lane,j)),0.f);
      uv[j]=__fadd_rn(a*decode(take_gate8528(uw,lane,j)),0.f);
    }
    float gd=reduce<W>(gv),ud=reduce<W>(uv);
    if(lane==0){
      size_t si=e*16*48+((offset+row)/128)*48+b;
      g+=gd*xs[token*48+b]*sg[si];
      u+=ud*xs[token*48+b]*su[si];
    }
  }
  if(lane==0){gate[route*256+row]=static_cast<__nv_bfloat16>(g);up[route*256+row]=static_cast<__nv_bfloat16>(u);}
}
__global__ void down(__nv_bfloat16* output,const uint8_t* x,const float* xs,const int64_t* ids,
 const uint8_t* weight,const float* scales,size_t routes,size_t offset){
  constexpr int W=32,V=4;
  size_t row=size_t(blockIdx.x)*8+threadIdx.x/W,route=blockIdx.y;
  if(row>=6144||route>=routes)return;
  unsigned lane=threadIdx.x%W;size_t e=size_t(ids[route]);float result=0.f;
  for(size_t b=0;b<2;++b){
    float values[V];
    const uint32_t xv=*reinterpret_cast<const uint32_t*>(x+route*256+b*128+4*lane);
    const uint32_t wv=*reinterpret_cast<const uint32_t*>(weight+(e*6144+row)*256+b*128+4*lane);
    #pragma unroll
    for(int j=0;j<V;++j){
      size_t k=b*128+lane+size_t(j)*W;
      values[j]=__fadd_rn(decode(take_down8528(xv,lane,j))*decode(take_down8528(wv,lane,j)),0.f);
    }
    float dot=reduce<W>(values);
    if(lane==0)result+=dot*xs[route*2+b]*scales[e*48*16+(row/128)*16+offset/128+b];
  }
  if(lane==0)output[route*6144+row]=static_cast<__nv_bfloat16>(result);
}
}
extern "C" int glm_vec_validate_r8528(const int64_t* ids,size_t count,size_t experts,int* status,void* stream){
  validate<<<unsigned((count+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(ids,count,experts,status);
  return int(cudaGetLastError());
}
namespace { thread_local int fragment_policy9048=0; thread_local uint64_t fragment_counts9048[2]={0,0}; }
extern "C" int glm_gate_fragment_policy_r9048(int mode) { if(mode<0||mode>1)return -1;fragment_policy9048=mode;return 0; }
extern "C" int glm_gate_fragment_counts_r9048(uint64_t*out,size_t n) { if(!out||n!=2)return -1;out[0]=fragment_counts9048[0];out[1]=fragment_counts9048[1];return 0; }
extern "C" int glm_vec_gate_r8528(void* gate,void* up,const void* x,const void* xs,const void* ids,
 const void* wg,const void* sg,const void* wu,const void* su,size_t routes,size_t offset,void* stream){
  ++fragment_counts9048[fragment_policy9048];
  if(fragment_policy9048) gate_up_register_mmac_r9048<<<dim3(4,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<__nv_bfloat16*>(gate),static_cast<__nv_bfloat16*>(up),static_cast<const uint8_t*>(x),
   static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(wg),
   static_cast<const float*>(sg),static_cast<const uint8_t*>(wu),static_cast<const float*>(su),routes,offset);
  else {  gate_up<<<dim3(64,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(
   static_cast<__nv_bfloat16*>(gate),static_cast<__nv_bfloat16*>(up),static_cast<const uint8_t*>(x),
   static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(wg),
   static_cast<const float*>(sg),static_cast<const uint8_t*>(wu),static_cast<const float*>(su),routes,offset);
  }
  return int(cudaGetLastError());
}
extern "C" int glm_vec_down_r8528(void* output,const void* x,const void* xs,const void* ids,
 const void* weight,const void* scales,size_t routes,size_t offset,void* stream){
  down<<<dim3(768,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(
   static_cast<__nv_bfloat16*>(output),static_cast<const uint8_t*>(x),static_cast<const float*>(xs),
   static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(weight),static_cast<const float*>(scales),routes,offset);
  return int(cudaGetLastError());
}
