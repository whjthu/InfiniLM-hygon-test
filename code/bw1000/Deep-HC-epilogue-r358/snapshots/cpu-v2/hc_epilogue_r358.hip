#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// Exact T1 production geometry. Prefix normalization/GEMM and the existing
// Sinkhorn remain Runtime operations; this replaces only their epilogue.
__device__ static uint16_t bf16_rn(float x) {
    uint32_t u=__float_as_uint(x);
    if ((u&0x7fffffffU)>0x7f800000U) return uint16_t((u>>16)|0x40U);
    u+=0x7fffU+((u>>16)&1U);return uint16_t(u>>16);
}
__device__ static float affine(float x,float scale,float base) {
    volatile float p=x*scale;return p+base;
}
__device__ static float sigmoid(float x) {return 1.f/(1.f+expf(-x));}

__global__ __launch_bounds__(256) void hc_epilogue_t1(
 const float* xf,const float* mixes,const float* base,const float* scale,
 const float* constants,uint16_t* collapsed,float* post,float* comb,unsigned* hit) {
    __shared__ float pre[4];
    unsigned lane=threadIdx.x;
    if(lane<4) {
        pre[lane]=sigmoid(affine(mixes[lane],scale[0],base[lane]))+constants[1];
        post[lane]=sigmoid(affine(mixes[4+lane],scale[1],base[4+lane]))*constants[2];
        float z[4];float maximum=-INFINITY;
        for(unsigned j=0;j<4;++j){z[j]=affine(mixes[8+4*lane+j],scale[2],base[8+4*lane+j]);maximum=fmaxf(maximum,z[j]);}
        float sum=0.f;
        for(unsigned j=0;j<4;++j){z[j]=expf(z[j]-maximum);sum+=z[j];}
        for(unsigned j=0;j<4;++j)comb[4*lane+j]=z[j]/sum+constants[1];
    }
    __syncthreads();
    for(unsigned i=lane;i<4096;i+=256) {
        float a=pre[0]*xf[i],b=pre[1]*xf[4096+i];
        float c=pre[2]*xf[8192+i],d=pre[3]*xf[12288+i];
        collapsed[i]=bf16_rn((a+b)+(c+d));
    }
    if(lane==0)atomicAdd(hit,1u);
}
extern "C" int deep_hc_epilogue_r358(const float* xf,const float* mixes,
 const float* base,const float* scale,const float* constants,void* collapsed,
 float* post,float* comb,unsigned* hit,void* stream) {
    if(!xf||!mixes||!base||!scale||!constants||!collapsed||!post||!comb||!hit)return int(hipErrorInvalidValue);
    hc_epilogue_t1<<<1,256,0,reinterpret_cast<hipStream_t>(stream)>>>(xf,mixes,base,scale,constants,static_cast<uint16_t*>(collapsed),post,comb,hit);
    return int(hipGetLastError());
}
