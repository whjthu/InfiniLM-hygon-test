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
__global__ void gate_up_lookahead9042(__nv_bfloat16* gate,__nv_bfloat16* up,const uint8_t* x,const float* xs,
 const int64_t* ids,const uint8_t* wg,const float* sg,const uint8_t* wu,const float* su,
 size_t routes,size_t offset){
  constexpr int W=64,V=2;
  size_t row=size_t(blockIdx.x)*4+threadIdx.x/W,route=blockIdx.y;
  if(row>=256||route>=routes)return;
  unsigned lane=threadIdx.x%W;size_t token=route/8,e=size_t(ids[route]);
  float g=0.f,u=0.f;
  uint32_t xv=*reinterpret_cast<const uint16_t*>(x+token*6144+2*lane);
  uint32_t gw=*reinterpret_cast<const uint16_t*>(wg+(e*256+row)*6144+2*lane);
  uint32_t uw=*reinterpret_cast<const uint16_t*>(wu+(e*256+row)*6144+2*lane);
  for(size_t b=0;b<48;++b){
    float gv[V],uv[V];
    uint32_t nx=0,ng=0,nu=0;
    if(b+1<48){
      nx=*reinterpret_cast<const uint16_t*>(x+token*6144+(b+1)*128+2*lane);
      ng=*reinterpret_cast<const uint16_t*>(wg+(e*256+row)*6144+(b+1)*128+2*lane);
      nu=*reinterpret_cast<const uint16_t*>(wu+(e*256+row)*6144+(b+1)*128+2*lane);
    }
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
    xv=nx;gw=ng;uw=nu;
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
// SOURCE ONLY: same owner thread, defaultOFF, no persistent GPU state.
namespace {thread_local int gate_lookahead9042=0;thread_local uint64_t gate_lookahead_calls9042[2]{};}
extern "C" int glm_gate_lookahead_policy_r9042(int mode){if(mode<0||mode>1)return -1;gate_lookahead9042=mode;return 0;}
extern "C" int glm_gate_lookahead_counts_r9042(uint64_t*out,size_t n){if(!out||n!=2)return -1;out[0]=gate_lookahead_calls9042[0];out[1]=gate_lookahead_calls9042[1];return 0;}
extern "C" int glm_vec_gate_r8528(void* gate,void* up,const void* x,const void* xs,const void* ids,
 const void* wg,const void* sg,const void* wu,const void* su,size_t routes,size_t offset,void* stream){
  ++gate_lookahead_calls9042[gate_lookahead9042];
  if(gate_lookahead9042){
  gate_up_lookahead9042<<<dim3(64,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(
   static_cast<__nv_bfloat16*>(gate),static_cast<__nv_bfloat16*>(up),static_cast<const uint8_t*>(x),
   static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(wg),
   static_cast<const float*>(sg),static_cast<const uint8_t*>(wu),static_cast<const float*>(su),routes,offset);
  }else{
  gate_up<<<dim3(64,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(
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
