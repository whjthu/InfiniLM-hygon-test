#include <cuda_runtime.h>
#include "topk_contract.hpp"
#include <cstdint>
namespace {
__global__ __launch_bounds__(256) void fixed_topk(const uint32_t*input,uint32_t*values,int64_t*indices){
 __shared__ uint64_t rank[deep416::PAD];
 for(int i=threadIdx.x;i<deep416::PAD;i+=blockDim.x)rank[i]=i<deep416::N?deep416::rank_key(input[i],i):uint64_t{0};
 __syncthreads();
 for(int span=2;span<=deep416::PAD;span<<=1)for(int stride=span>>1;stride;stride>>=1){
  for(int i=threadIdx.x;i<deep416::PAD;i+=blockDim.x){int j=i^stride;if(j<=i)continue;bool descending=(i&span)==0;
   auto a=rank[i],b=rank[j];if(descending?b>a:a>b){rank[i]=b;rank[j]=a;}}
  __syncthreads();
 }
 for(int i=threadIdx.x;i<deep416::K;i+=blockDim.x){int id=int(0xffffffffu-uint32_t(rank[i]));indices[i]=id;values[i]=input[id];}
}
}
extern "C" int deep_fixed_topk_launch_r416(const void*input,void*values,void*indices,void*raw_stream){
 if(!input||!values||!indices)return int(cudaErrorInvalidValue);
 auto stream=static_cast<cudaStream_t>(raw_stream);
 // Poison is production candidate cost, not an untimed stale-output exemption.
 auto rc=cudaMemsetAsync(values,0xff,deep416::K*sizeof(float),stream);if(rc!=cudaSuccess)return int(rc);
 rc=cudaMemsetAsync(indices,0xff,deep416::K*sizeof(int64_t),stream);if(rc!=cudaSuccess)return int(rc);
 fixed_topk<<<1,deep416::BLOCK,0,stream>>>(static_cast<const uint32_t*>(input),static_cast<uint32_t*>(values),static_cast<int64_t*>(indices));
 return int(cudaGetLastError());
}
