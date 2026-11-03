#include <cuda_runtime.h>
#include "topk_contract.hpp"
#include <cstdint>
namespace {
__global__ __launch_bounds__(256) void fixed_topk(const uint32_t*input,uint32_t*values,int64_t*indices){
 __shared__ uint32_t score[deep415::PAD];__shared__ int id[deep415::PAD];
 for(int i=threadIdx.x;i<deep415::PAD;i+=blockDim.x){score[i]=i<deep415::N?input[i]:0xff800000u;id[i]=i;}
 __syncthreads();
 for(int span=2;span<=deep415::PAD;span<<=1)for(int stride=span>>1;stride;stride>>=1){
  for(int i=threadIdx.x;i<deep415::PAD;i+=blockDim.x){int j=i^stride;if(j<=i)continue;bool descending=(i&span)==0;
   bool exchange=descending?deep415::better(score[j],id[j],score[i],id[i]):deep415::better(score[i],id[i],score[j],id[j]);
   if(exchange){auto b=score[i];auto index=id[i];score[i]=score[j];id[i]=id[j];score[j]=b;id[j]=index;}}
  __syncthreads();
 }
 for(int i=threadIdx.x;i<deep415::K;i+=blockDim.x){indices[i]=id[i];values[i]=input[id[i]];}
}
}
extern "C" int deep_fixed_topk_launch_r415(const void*input,void*values,void*indices,void*raw_stream){
 if(!input||!values||!indices)return int(cudaErrorInvalidValue);
 auto stream=static_cast<cudaStream_t>(raw_stream);
 // Poison is production candidate cost, not an untimed stale-output exemption.
 auto rc=cudaMemsetAsync(values,0xff,deep415::K*sizeof(float),stream);if(rc!=cudaSuccess)return int(rc);
 rc=cudaMemsetAsync(indices,0xff,deep415::K*sizeof(int64_t),stream);if(rc!=cudaSuccess)return int(rc);
 fixed_topk<<<1,deep415::BLOCK,0,stream>>>(static_cast<const uint32_t*>(input),static_cast<uint32_t*>(values),static_cast<int64_t*>(indices));
 return int(cudaGetLastError());
}
