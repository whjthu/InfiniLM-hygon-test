#include "contract.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <limits.h>
#include <array>
#include <atomic>
namespace{std::array<std::array<std::atomic<uint64_t>,4>,8> counters6054{};}
extern "C" int kimi401_decoder_counts(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=4)return 1;for(size_t i=0;i<4;++i)out[i]=counters6054[device][i].load();return 0;}
#include "native/cuda/ops/fused_moe_mxfp4/mxfp4_math.cuh"
// Exact original decode+BF16 rounding+F32 materialization, bounded u32 indexing.
template<uint32_t PER> __global__ void decode_static_r6054(const unsigned char*p,const unsigned char*s,const int*ids,__nv_bfloat16*out,uint32_t count){constexpr uint32_t per=PER;
 uint32_t total=count*per;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=gridDim.x*blockDim.x){uint32_t local=i/per,within=i%per,source=uint32_t(ids[local])*per+within;float a,b;mxfp4DecodePair(p[source],s[source/16],a,b);out[2*i]=__float2bfloat16_rn(a);out[2*i+1]=__float2bfloat16_rn(b);}
}
extern "C" int kimi401_bounded_decode_bf16_r6054(const void*p,const void*s,const void*ids,void*out,size_t count,size_t rows,size_t columns,void*stream){
 if(!p||!s||!ids||!out||!count||count>64||!((rows==768&&columns==3584)||(rows==3584&&columns==384)||(rows==384&&columns==3584)||(rows==3584&&columns==192)))return 1;
 size_t per=rows*(columns/2);if(!per||per>UINT_MAX/896||count*per>UINT_MAX/2||columns%32)return 1;
 size_t n=count*per;unsigned blocks=unsigned((n+255)/256);if(blocks>65535)blocks=65535;
 if(rows==768&&columns==3584)decode_static_r6054<1376256><<<blocks,256,0,(cudaStream_t)stream>>>((const unsigned char*)p,(const unsigned char*)s,(const int*)ids,(__nv_bfloat16*)out,(uint32_t)count);
 else if(rows==3584&&columns==384)decode_static_r6054<688128><<<blocks,256,0,(cudaStream_t)stream>>>((const unsigned char*)p,(const unsigned char*)s,(const int*)ids,(__nv_bfloat16*)out,(uint32_t)count);
 else if(rows==384&&columns==3584)decode_static_r6054<688128><<<blocks,256,0,(cudaStream_t)stream>>>((const unsigned char*)p,(const unsigned char*)s,(const int*)ids,(__nv_bfloat16*)out,(uint32_t)count);
 else if(rows==3584&&columns==192)decode_static_r6054<344064><<<blocks,256,0,(cudaStream_t)stream>>>((const unsigned char*)p,(const unsigned char*)s,(const int*)ids,(__nv_bfloat16*)out,(uint32_t)count);
 else return 1;int status=int(cudaGetLastError());if(!status){int dev=-1;if(cudaGetDevice(&dev)!=cudaSuccess||dev<0||dev>=8)return 1;auto&c=counters6054[dev];size_t offset=kimi401::role(rows,columns);c[offset].fetch_add(1);c[offset+1].fetch_add(count);}return status;
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace {
__global__ void gather_bf16(const __nv_bfloat16*x,const long long*rows,__nv_bfloat16*out,size_t count,size_t hidden){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count*hidden;i+=size_t(gridDim.x)*blockDim.x)out[i]=x[size_t(rows[i/hidden])*hidden+i%hidden];
}
}
extern "C" int kimi401_grouped_gather_bf16_r26(const void*x,const void*ids,void*out,size_t rows,size_t hidden,void*stream){
 size_t count=rows*hidden;unsigned grid=unsigned((count+255)/256>65535?65535:(count+255)/256);
 gather_bf16<<<grid,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const long long*>(ids),static_cast<__nv_bfloat16*>(out),rows,hidden);
 return int(cudaGetLastError());
}
