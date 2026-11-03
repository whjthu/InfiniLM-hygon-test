#include <cuda_runtime.h>
#include "router_math.hpp"
namespace {
__global__ __launch_bounds__(256) void suffix(const float*logits,const float*bias,const float*epsilon,const float*factor,float*weights,int64_t*ids,float*debug_scores,float*debug_selection){
 __shared__ uint64_t scratch[256];__shared__ float scores[256],selected[6];__shared__ int chosen;
 const int t=threadIdx.x;float s=deep419::score(logits[t]),v=s+bias[t];scores[t]=s;
 if(debug_scores)debug_scores[t]=s;if(debug_selection)debug_selection[t]=v;
 uint64_t rank=deep419::rank_key(v,t);__syncthreads();
 for(int k=0;k<6;++k){scratch[t]=rank;__syncthreads();
  for(int stride=128;stride;stride>>=1){if(t<stride){auto other=scratch[t+stride];if(other>scratch[t])scratch[t]=other;}__syncthreads();}
  if(t==0){chosen=int(0xffffffffu-uint32_t(scratch[0]));ids[k]=chosen;selected[k]=scores[chosen];}__syncthreads();
  if(t==chosen)rank=0;
 }
 if(t<6)weights[t]=deep419::normalize(selected,*epsilon,*factor,t);
}
}
extern "C" int deep_router_suffix_launch_r419(const void*logits,const void*bias,const void*epsilon,const void*factor,void*weights,void*ids,void*debug_scores,void*debug_selection,void*raw_stream){
 if(!logits||!bias||!epsilon||!factor||!weights||!ids)return int(cudaErrorInvalidValue);
 auto stream=static_cast<cudaStream_t>(raw_stream);auto rc=cudaMemsetAsync(weights,0xff,6*sizeof(float),stream);if(rc!=cudaSuccess)return int(rc);
 rc=cudaMemsetAsync(ids,0xff,6*sizeof(int64_t),stream);if(rc!=cudaSuccess)return int(rc);
 suffix<<<1,256,0,stream>>>(static_cast<const float*>(logits),static_cast<const float*>(bias),static_cast<const float*>(epsilon),static_cast<const float*>(factor),static_cast<float*>(weights),static_cast<int64_t*>(ids),static_cast<float*>(debug_scores),static_cast<float*>(debug_selection));
 return int(cudaGetLastError());
}
