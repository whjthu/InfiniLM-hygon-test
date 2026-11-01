#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// Fixed current T1/H8/D128 path. Decode arithmetic and reduction trees match
// Hygon IndexerScores c9fb2fbb...; query decode is shared across all keys.
__device__ static float decode_r359(unsigned char b) {
    unsigned e=(b>>3)&15,m=b&7;
    float value=e?ldexpf(1.f+float(m)/8,int(e)-7):ldexpf(float(m),-9);
    return b&128?-value:value;
}
__device__ static const unsigned char* key_r359(const unsigned char*pages,
 const unsigned char*tail,size_t slot_base,size_t before,size_t tail_begin,size_t key) {
    if(key>=before)return tail+(tail_begin+key-before)*128;
    size_t slot=slot_base+key;
    return pages+(slot/64)*8448+(slot%64)*128;
}
__device__ static float scale_r359(const unsigned char*pages,const float*tail,
 size_t slot_base,size_t before,size_t tail_begin,size_t key) {
    if(key>=before)return tail[tail_begin+key-before];
    size_t slot=slot_base+key;
    return *reinterpret_cast<const float*>(pages+(slot/64)*8448+8192+(slot%64)*4);
}
__global__ void query_r359(const unsigned char*q,const float*w,const int32_t*v,
 float*decoded,size_t keys,int*status) {
    if(threadIdx.x==0 && (v[0]<0||size_t(v[0])>keys))atomicOr(status,1);
    for(size_t i=threadIdx.x;i<8*128;i+=blockDim.x) {
        if(v[0]>0) {
            if((q[i]&127)==127)atomicOr(status,2);
            if(i%128==0&&!isfinite(w[i/128]))atomicOr(status,2);
            decoded[i]=decode_r359(q[i]);
        }
    }
}
__global__ void validate_keys_r359(const unsigned char*pages,
 const unsigned char*tail,const float*tail_scales,const int32_t*v,
 size_t before,size_t slot_base,size_t tail_begin,size_t keys,int*status) {
    if(v[0]<0||size_t(v[0])>keys)return;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(v[0])*128;
        i+=size_t(gridDim.x)*blockDim.x) {
        size_t key=i/128,d=i%128;
        if((key_r359(pages,tail,slot_base,before,tail_begin,key)[d]&127)==127)atomicOr(status,2);
        if(d==0) {
            float scale=scale_r359(pages,tail_scales,slot_base,before,tail_begin,key);
            if(!isfinite(scale)||scale<=0)atomicOr(status,2);
        }
    }
}
__global__ __launch_bounds__(128) void scores_r359(float*out,
 const float*q,const float*w,const unsigned char*pages,const unsigned char*tail,
 const float*tail_scales,const int32_t*v,size_t before,size_t slot_base,
 size_t tail_begin,size_t keys,const int*status) {
    if(*status)return;
    __shared__ float key[128],partial[128],contribution[128];
    unsigned lane=threadIdx.x%16,head=threadIdx.x/16;
    for(size_t ki=blockIdx.x;ki<keys;ki+=gridDim.x) {
        if(ki>=size_t(v[0])){if(threadIdx.x==0)out[ki]=-INFINITY;continue;}
        key[threadIdx.x]=decode_r359(key_r359(pages,tail,slot_base,before,tail_begin,ki)[threadIdx.x]);
        contribution[threadIdx.x]=0;__syncthreads();
        float dot=0;
        for(unsigned d=lane;d<128;d+=16)dot=__fadd_rn(dot,__fmul_rn(q[head*128+d],key[d]));
        partial[threadIdx.x]=dot;__syncthreads();
        for(unsigned span=8;span;span>>=1) {
            if(lane<span)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+span]);
            __syncthreads();
        }
        if(lane==0)contribution[head]=__fmul_rn(
            fmaxf(__fmul_rn(partial[threadIdx.x],scale_r359(pages,tail_scales,slot_base,before,tail_begin,ki)),0.f),w[head]);
        __syncthreads();
        for(unsigned span=64;span;span>>=1) {
            if(threadIdx.x<span)contribution[threadIdx.x]=__fadd_rn(contribution[threadIdx.x],contribution[threadIdx.x+span]);
            __syncthreads();
        }
        if(threadIdx.x==0)out[ki]=contribution[0];
        __syncthreads();
    }
}
extern "C" int deep_paged_scores_r359(void*out,const void*raw,const void*q,
 const void*w,const void*tail,const void*tail_scales,const void*visible,
 void*status,void*decoded,size_t before,size_t keys,size_t slot_base,
 size_t raw_slots,size_t tail_begin,void*stream) {
    if(!out||!raw||!q||!w||!visible||!status||!decoded||!keys||keys>8192||
       before>keys||keys-before>1||slot_base>raw_slots||before>raw_slots-slot_base||
       (keys>before&&(!tail||!tail_scales)))return int(hipErrorInvalidValue);
    auto st=reinterpret_cast<hipStream_t>(stream);auto flag=static_cast<int*>(status);
    auto error=hipMemsetAsync(flag,0,4,st);if(error!=hipSuccess)return int(error);
    query_r359<<<1,256,0,st>>>(static_cast<const unsigned char*>(q),static_cast<const float*>(w),static_cast<const int32_t*>(visible),static_cast<float*>(decoded),keys,flag);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    validate_keys_r359<<<32,256,0,st>>>(static_cast<const unsigned char*>(raw),static_cast<const unsigned char*>(tail),static_cast<const float*>(tail_scales),static_cast<const int32_t*>(visible),before,slot_base,tail_begin,keys,flag);
    error=hipGetLastError();if(error!=hipSuccess)return int(error);
    scores_r359<<<unsigned(keys),128,0,st>>>(static_cast<float*>(out),static_cast<const float*>(decoded),static_cast<const float*>(w),static_cast<const unsigned char*>(raw),static_cast<const unsigned char*>(tail),static_cast<const float*>(tail_scales),static_cast<const int32_t*>(visible),before,slot_base,tail_begin,keys,flag);
    return int(hipGetLastError());
}
