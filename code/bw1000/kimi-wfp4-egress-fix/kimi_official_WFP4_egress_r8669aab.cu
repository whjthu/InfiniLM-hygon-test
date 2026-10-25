// Mode114 specialization derived from HYGON-AI/Moe commit46ae5db (MIT).
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Raw FP32 W13 egress, original SITU and original I192 W2; no BF16 expertbank.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
namespace infini::ops::kimi_i192_r8669y {
__global__ void packRoutesI192R8669y(const int32_t*,int32_t*,int32_t*,size_t,size_t);
template<class T>__global__ void staticW2I192R8669y(T*,const T*,const int32_t*,const float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t,size_t);
}
namespace {
using B=__nv_bfloat16;using B4=unsigned short __attribute__((vector_size(8)));using B8=unsigned short __attribute__((vector_size(16)));using F4=float __attribute__((vector_size(16)));
constexpr unsigned E=896,H=3584,I=192,N=384,R=256;
__global__ void sorted16_r8669zu(const int*counts,const int*routes,int*sorted,int*experts,int*num){
 if(threadIdx.x||blockIdx.x)return;int g=0;
 for(unsigned e=0;e<E;++e)for(int p=0;p<counts[e];p+=16){
  if(g>=256){*num=-1;return;}experts[g]=int(e);
  for(int j=0;j<16;++j)sorted[g*16+j]=(p+j<counts[e])?routes[e*R+p+j]:int(R);++g;
 }*num=g;
}
// Source constants: M16/N64/K64, WARP_M16/N32/K64,2waves,FP4_E2M1,
// INTERLEAVEfalse,DEQUANT_16BITfalse,PACK_SCALES8,FP4_K_SPLITfalse.
__global__ void mode114_F32_r8669zu(float*out,const B*x,const uint32_t*w,const uint8_t*s,const int*sorted,const int*experts,const int*groups){
 const int group=blockIdx.z;if(group>=*groups||sorted[group*16]>=int(R))return;
 const unsigned tid=threadIdx.x,warp=tid/64,lane=tid%64,rowid=lane%16,colid=lane/16;
 const unsigned br=tid/2,bc=tid%2,nb=blockIdx.x*64;
 const unsigned e=unsigned(experts[group]);
 extern __shared__ unsigned short shared[];auto*A=shared;auto*Bsh=A+16*64;
 F4 accum[2]={{},{}};
 // Same traversal and FP32 accumulator sequence as official stage1 helper.
 for(unsigned ko=0;ko<H;ko+=512){
  uint8_t scales[8];
  #pragma unroll
  for(unsigned j=0;j<8;++j)scales[j]=s[(size_t(e)*N+nb+br)*(H/32)+ko/32+j*2+bc];
  #pragma unroll
  for(unsigned ki=0;ki<512;ki+=64){
   const unsigned k=ko+ki;
   if(warp==0){
    #pragma unroll
    for(unsigned ld=0;ld<2;++ld){const unsigned ar=ld*8+lane/8,ac=tid%8;const unsigned route=unsigned(sorted[group*16+ar]);const unsigned token=(route<R?route:R-1)/16;
     *reinterpret_cast<B8*>(A+ar*64+((ar+ac)%8)*8)=*reinterpret_cast<const B8*>(x+token*H+k+ac*8);
    }
   }
   const auto*q=w+(size_t(e)*N+nb+br)*(H/8)+(k+bc*32)/8;
   #pragma unroll
   for(unsigned word=0;word<4;++word){const uint32_t qw=q[word];B8 v;const unsigned c=bc*4+word;
    #pragma unroll
    for(unsigned p=0;p<4;++p){
     // Keep original E2M1/scale endpoint semantics, including0 and255.
     float lo,hi;const uint8_t adjacent=uint8_t(((qw>>(p*4))&15)|(((qw>>(16+p*4))&15)<<4));
     ::mxfp4DecodePair(adjacent,scales[ki/64],lo,hi);
     v[p*2]=__bfloat16_as_ushort(__float2bfloat16_rn(lo));v[p*2+1]=__bfloat16_as_ushort(__float2bfloat16_rn(hi));
    }
    *reinterpret_cast<B8*>(Bsh+br*64+((br+c)%8)*8)=v;
   }
   __syncthreads();
   #pragma unroll
   for(unsigned ks=0;ks<64;ks+=32){const unsigned kc=ks/8+colid;B8 av=*reinterpret_cast<const B8*>(A+rowid*64+((rowid+kc)%8)*8);
    #pragma unroll
    for(unsigned nt=0;nt<2;++nt){const unsigned nr=warp*32+nt*16+rowid;B8 bv=*reinterpret_cast<const B8*>(Bsh+nr*64+((nr+kc)%8)*8);auto*aa=reinterpret_cast<B4*>(&av);auto*bb=reinterpret_cast<B4*>(&bv);
     accum[nt]=__builtin_hcu_mmac_f32_16x16x16_bf16(aa[0],bb[0],accum[nt]);accum[nt]=__builtin_hcu_mmac_f32_16x16x16_bf16(aa[1],bb[1],accum[nt]);
    }
   }__syncthreads();
  }
 }
 const unsigned route=unsigned(sorted[group*16+rowid]);if(route<R){
  #pragma unroll
  for(unsigned nt=0;nt<2;++nt){
   #pragma unroll
   for(unsigned ro=0;ro<2;++ro){
    #pragma unroll
    for(unsigned ri=0;ri<2;++ri){const unsigned reg=ro*2+ri,col=nb+warp*32+nt*16+ro*8+ri*4+colid;out[size_t(route)*N+col]=accum[nt][reg];}
   }
  }
 }
}
__global__ void Situ_r8669zu(B*activated,const float*raw){unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=R*I)return;unsigned r=j/I,i=j%I;activated[j]=::mxfp4Store<B>(infini::ops::mxfp4_detail::fusedMoeMxfp4Activate(raw[r*N+i],raw[r*N+I+i],size_t{2}));}
}
extern "C" int modern_kimi_official_WFP4_r8669zu(int device,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*extra,void*stream){
 int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=device||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!extra)return int(cudaErrorInvalidValue);
 auto st=static_cast<cudaStream_t>(stream);auto*raw=static_cast<float*>(extra);auto*sorted=reinterpret_cast<int*>(raw+R*N);auto*experts=sorted+R*16;auto*num=experts+R;
 auto status=cudaMemsetAsync(counts,0,E*sizeof(int),st);if(status!=cudaSuccess)return int(status);
 infini::ops::kimi_i192_r8669y::packRoutesI192R8669y<<<1,256,0,st>>>(static_cast<const int*>(ids),static_cast<int*>(counts),static_cast<int*>(routes),E,16);
 sorted16_r8669zu<<<1,1,0,st>>>(static_cast<const int*>(counts),static_cast<const int*>(routes),sorted,experts,num);
 mode114_F32_r8669zu<<<dim3(6,1,256),128,16448,st>>>(raw,static_cast<const B*>(input),static_cast<const uint32_t*>(w13),static_cast<const uint8_t*>(s13),sorted,experts,num);
 status=cudaGetLastError();if(status!=cudaSuccess)return int(status);
 Situ_r8669zu<<<(R*I+255)/256,256,0,st>>>(static_cast<B*>(activation),raw);
 infini::ops::kimi_i192_r8669y::staticW2I192R8669y<B><<<H*16,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),16,size_t{16},E,H,I,size_t{0});
 return int(cudaGetLastError());
}
