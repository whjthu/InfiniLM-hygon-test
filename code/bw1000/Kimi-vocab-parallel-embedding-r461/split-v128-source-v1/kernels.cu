#include "kernels.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
namespace {
template<class I>__global__ void store(unsigned short*c,const unsigned short*k,const unsigned short*v,const I*slots,size_t t,long ks0,long ks1,long vs0,long vs1){
 size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=t*3*320)return;
 size_t row=z/(3*320),h=z/320%3,d=z%320;long slot=slots[row];if(slot<0)return;assert(slot<132*256);
 c[(slot*3+h)*320+d]=d<192?k[row*ks0+h*ks1+d]:v[row*vs0+h*vs1+d-192];
}
template<class I>__global__ void gather(unsigned short*out,const unsigned short*c,const I*table,size_t start,size_t count,size_t h,size_t dim){
 size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=count*dim)return;
 size_t pos=start+z/dim;long page=table[pos/256];assert(page>=0&&page<132);
 size_t feature=z%dim+(dim==128?192:0);out[z]=c[((page*256+pos%256)*3+h)*320+feature];
}
__global__ void mask(float*s,size_t rows,size_t total,size_t past,size_t begin){
 size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=rows*total)return;
 if(z%total>past+begin+z/total)s[z]=-INFINITY;
}
}
extern "C" int kimi_split_store(void*c,const void*k,const void*v,const void*slots,int bits,size_t t,const int64_t*ks,const int64_t*vs,void*stream){
 if(!c||!k||!v||!slots||!ks||!vs||!t||t>512||(bits!=32&&bits!=64))return 1;
 auto s=static_cast<cudaStream_t>(stream);auto blocks=(t*3*320+255)/256;
 if(bits==32)store<<<blocks,256,0,s>>>((unsigned short*)c,(const unsigned short*)k,(const unsigned short*)v,(const int32_t*)slots,t,ks[0],ks[1],vs[0],vs[1]);
 else store<<<blocks,256,0,s>>>((unsigned short*)c,(const unsigned short*)k,(const unsigned short*)v,(const int64_t*)slots,t,ks[0],ks[1],vs[0],vs[1]);
 return int(cudaGetLastError());
}
extern "C" int kimi_split_gather(void*out,const void*c,const void*table,int bits,size_t start,size_t count,size_t h,size_t dim,void*stream){
 if(!out||!c||!table||(bits!=32&&bits!=64)||h>=3||(dim!=192&&dim!=128)||!count||count>256||start+count>32768)return 1;
 auto s=static_cast<cudaStream_t>(stream);auto blocks=(count*dim+255)/256;
 if(bits==32)gather<<<blocks,256,0,s>>>((unsigned short*)out,(const unsigned short*)c,(const int32_t*)table,start,count,h,dim);
 else gather<<<blocks,256,0,s>>>((unsigned short*)out,(const unsigned short*)c,(const int64_t*)table,start,count,h,dim);
 return int(cudaGetLastError());
}
extern "C" int kimi_split_mask(void*out,size_t rows,size_t total,size_t past,size_t begin,void*stream){
 if(!out||!rows||rows>8||!total||total>32768||past+begin+rows>total)return 1;
 mask<<<(rows*total+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>((float*)out,rows,total,past,begin);return int(cudaGetLastError());
}
