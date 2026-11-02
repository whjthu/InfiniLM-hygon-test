#pragma once
#include <stdint.h>
#include <math.h>
#ifdef __HIPCC__
#define HC372_INLINE __device__ __forceinline__
#else
#include <cstring>
#define HC372_INLINE inline
#endif
namespace hc372 {
HC372_INLINE float unpack(uint16_t value){
 uint32_t word=uint32_t(value)<<16;
#ifdef __HIPCC__
 return __uint_as_float(word);
#else
 float out;std::memcpy(&out,&word,4);return out;
#endif
}
HC372_INLINE uint16_t pack(float value){
 uint32_t word;
#ifdef __HIPCC__
 word=__float_as_uint(value);
#else
 std::memcpy(&word,&value,4);
#endif
 if((word&0x7fffffffU)>0x7f800000U)return uint16_t((word>>16)|0x40U);
 word+=0x7fffU+((word>>16)&1U);return uint16_t(word>>16);
}
HC372_INLINE float affine(float value,float scale,float base){volatile float product=value*scale;return product+base;}
HC372_INLINE float sigmoid(float value){return 1.f/(1.f+expf(-value));}
HC372_INLINE void row_weights(unsigned row,const float*mixes,const float*base,const float*scale,const float*constants,float*pre,float*post,float*comb){
 pre[row]=sigmoid(affine(mixes[row],scale[0],base[row]))+constants[1];
 post[row]=sigmoid(affine(mixes[4+row],scale[1],base[4+row]))*constants[2];
 float z[4],maximum=-INFINITY;
 for(unsigned j=0;j<4;++j){z[j]=affine(mixes[8+4*row+j],scale[2],base[8+4*row+j]);maximum=fmaxf(maximum,z[j]);}
 float sum=0.f;for(unsigned j=0;j<4;++j){z[j]=expf(z[j]-maximum);sum+=z[j];}
 for(unsigned j=0;j<4;++j)comb[4*row+j]=z[j]/sum+constants[1];
}
// Literal r89 alternating normalization body: row pairwise sum; column
// left-associated sum; every epsilon addition and FP32 division retained.
HC372_INLINE void sinkhorn(float*a,float eps,int iterations){
 for(int iteration=0;iteration<iterations;++iteration){
  if(iteration){
   for(int row=0;row<4;++row){float sum=(a[4*row]+a[4*row+1])+(a[4*row+2]+a[4*row+3]);float denominator=sum+eps;for(int col=0;col<4;++col)a[4*row+col]=a[4*row+col]/denominator;}
  }
  for(int col=0;col<4;++col){float sum=((a[col]+a[4+col])+a[8+col])+a[12+col];float denominator=sum+eps;for(int row=0;row<4;++row)a[4*row+col]=a[4*row+col]/denominator;}
 }
}
HC372_INLINE uint16_t collapse_at(const float*xf,const float*pre,unsigned i){
 float a=pre[0]*xf[i],b=pre[1]*xf[4096+i],c=pre[2]*xf[8192+i],d=pre[3]*xf[12288+i];
 return pack((a+b)+(c+d));
}
}
