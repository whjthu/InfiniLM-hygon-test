#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
namespace {
__device__ __forceinline__ float decode8508(uint8_t v){
 unsigned e=(v>>3)&15,m=v&7,sign=unsigned(v&128)<<24;
 if(!e)return __uint_as_float(__float_as_uint(float(m)*0.001953125f)|sign);
 return __uint_as_float(sign|(e==15&&m==7?0x7fc00000u:((e+120)<<23)|(m<<20)));
}
// One wave, one route, sixteen output channels. Repeated rows deliberately
// avoid GPU->CPU expert grouping. Only row0 is useful; no gain is assumed.
template<bool Gate> __global__ void matrix8508(
 __nv_bfloat16*out,__nv_bfloat16*outup,const uint8_t*x,const float*xs,
 const int64_t*ids,const uint8_t*w,const float*ws,const uint8_t*wu,const float*su,
 size_t routes,size_t offset){
 constexpr int K=Gate?6144:256,B=K/128,N=Gate?512:6144;
 const size_t route=blockIdx.y,nbase=size_t(blockIdx.x)*16;
 const unsigned t=threadIdx.x;const size_t expert=size_t(ids[route]),token=Gate?route/8:route;
 __shared__ __align__(32) __nv_bfloat16 aa[16*128];
 __shared__ __align__(32) __nv_bfloat16 bb[128*16];
 __shared__ __align__(32) float cc[16*16];
 float sum=0.f;
 using namespace nvcuda;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bf;
 wmma::fragment<wmma::accumulator,16,16,16,float> accum;
 for(int b=0;b<B;++b){
  for(unsigned j=t;j<16*128;j+=64){
   aa[j]=static_cast<__nv_bfloat16>(decode8508(x[token*K+b*128+j%128]));
   const unsigned col=j/128,k=j%128;const size_t n=nbase+col;
   const uint8_t*weight=Gate&&n>=256?wu:w;const size_t wn=Gate?n%256:n;
   bb[k*16+col]=static_cast<__nv_bfloat16>(decode8508(weight[(expert*(Gate?256:6144)+wn)*K+b*128+k]));
  }
  __syncthreads();wmma::fill_fragment(accum,0.f);
  #pragma unroll
  for(int kk=0;kk<128;kk+=16){
   wmma::load_matrix_sync(af,aa+kk,128);wmma::load_matrix_sync(bf,bb+kk*16,16);
   wmma::mma_sync(accum,af,bf,accum);
  }
  wmma::store_matrix_sync(cc,accum,16,wmma::mem_row_major);__syncthreads();
  if(t<16){
   const size_t n=nbase+t,wn=Gate?n%256:n;const float*scale=Gate&&n>=256?su:ws;
   size_t si=Gate?expert*16*48+((offset+wn)/128)*48+b:expert*48*16+(n/128)*16+offset/128+b;
   sum+=cc[t]*xs[token*B+b]*scale[si];
  }
  __syncthreads();
 }
 if(t<16){size_t n=nbase+t;if(Gate){__nv_bfloat16*target=n>=256?outup:out;target[route*256+n%256]=static_cast<__nv_bfloat16>(sum);}else out[route*N+n]=static_cast<__nv_bfloat16>(sum);}
}
}
extern "C" int glm_matrix_gate_r8508(void*g,void*u,const void*x,const void*xs,const void*ids,const void*w,const void*ws,const void*wu,const void*su,size_t routes,size_t offset,void*s){
 if(!g||!u||!x||!xs||!ids||!w||!ws||!wu||!su||!routes||routes>1024||routes%8||offset>1792||offset%256)return -1;
 matrix8508<true><<<dim3(32,unsigned(routes)),64,0,static_cast<cudaStream_t>(s)>>>(static_cast<__nv_bfloat16*>(g),static_cast<__nv_bfloat16*>(u),static_cast<const uint8_t*>(x),static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(w),static_cast<const float*>(ws),static_cast<const uint8_t*>(wu),static_cast<const float*>(su),routes,offset);
 return int(cudaGetLastError());
}
extern "C" int glm_matrix_down_r8508(void*out,const void*x,const void*xs,const void*ids,const void*w,const void*ws,size_t routes,size_t offset,void*s){
 if(!out||!x||!xs||!ids||!w||!ws||!routes||routes>1024||routes%8||offset>1792||offset%256)return -1;
 matrix8508<false><<<dim3(384,unsigned(routes)),64,0,static_cast<cudaStream_t>(s)>>>(static_cast<__nv_bfloat16*>(out),nullptr,static_cast<const uint8_t*>(x),static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(w),static_cast<const float*>(ws),nullptr,nullptr,routes,offset);
 return int(cudaGetLastError());
}
