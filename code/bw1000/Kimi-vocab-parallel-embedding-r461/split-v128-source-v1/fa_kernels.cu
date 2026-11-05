#include "kernels.cu" // one private GPU TU: same exact cache writer, no SDK change
#include "fa_kernels.h"
#include "merge_math.hpp"
#include <cuda_bf16.h>
namespace {
template<class I>__global__ void pack_fa(unsigned short*k,unsigned short*v,int*len,const unsigned short*c,const I*table,size_t start,size_t count){
 size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z==0)*len=int(count);if(z>=count*3*192)return;
 size_t pos=start+z/(3*192),h=z/192%3,d=z%192;long page=table[pos/256];assert(page>=0&&page<132);
 size_t base=((page*256+pos%256)*3+h)*320;k[z]=c[base+d];v[z]=d<128?c[base+192+d]:0;
}
__global__ void factors(float*state,float*f,const float*lse,size_t rows,bool first){
 size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=rows*3)return;
 size_t row=z/3,h=z%3;float b=lse[h*rows+row];assert(isfinite(b));
 auto values=kimi_split_v128::factors(first?0.f:state[z*2],first?0.f:state[z*2+1],b,first);f[z*2]=values.a;f[z*2+1]=values.b;state[z*2]=values.maximum;state[z*2+1]=values.denominator;
}
__global__ void merge(float*sum,const __nv_bfloat16*partial,const float*f,size_t rows,bool first){
 size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=rows*3*128)return;
 size_t headrow=z/128,d=z%128;float value=__bfloat162float(partial[headrow*192+d]);
 sum[z]=kimi_split_v128::merge_value(first?0.f:sum[z],value,f[headrow*2],f[headrow*2+1],first);
}
__global__ void finish(__nv_bfloat16*out,const float*sum,const float*state,size_t rows,size_t first){
 size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=rows*3*128)return;
 float den=state[(z/128)*2+1];assert(den>0&&isfinite(den));out[first*3*128+z]=__float2bfloat16(sum[z]/den);
}
}
extern "C" int kimi_split_fa_pack(void*k,void*v,void*len,const void*c,const void*table,int bits,size_t start,size_t count,void*stream){
 if(!k||!v||!len||!c||!table||(bits!=32&&bits!=64)||!count||count>3072||start+count>32768)return 1;
 auto s=static_cast<cudaStream_t>(stream);size_t blocks=(count*3*192+255)/256;
 if(bits==32)pack_fa<<<blocks,256,0,s>>>((unsigned short*)k,(unsigned short*)v,(int*)len,(const unsigned short*)c,(const int32_t*)table,start,count);
 else pack_fa<<<blocks,256,0,s>>>((unsigned short*)k,(unsigned short*)v,(int*)len,(const unsigned short*)c,(const int64_t*)table,start,count);
 return int(cudaGetLastError());
}
extern "C" int kimi_split_fa_factors(void*s,void*f,const void*l,size_t rows,int first,void*stream){
 if(!s||!f||!l||!rows||rows>32||(first!=0&&first!=1))return 1;
 factors<<<(rows*3+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>((float*)s,(float*)f,(const float*)l,rows,first);return int(cudaGetLastError());
}
extern "C" int kimi_split_fa_merge(void*s,const void*p,const void*f,size_t rows,int first,void*stream){
 if(!s||!p||!f||!rows||rows>32||(first!=0&&first!=1))return 1;
 merge<<<(rows*3*128+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>((float*)s,(const __nv_bfloat16*)p,(const float*)f,rows,first);return int(cudaGetLastError());
}
extern "C" int kimi_split_fa_finish(void*out,const void*sum,const void*state,size_t rows,size_t first,void*stream){
 if(!out||!sum||!state||!rows||rows>32||first+rows>512)return 1;
 finish<<<(rows*3*128+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>((__nv_bfloat16*)out,(const float*)sum,(const float*)state,rows,first);return int(cudaGetLastError());
}
