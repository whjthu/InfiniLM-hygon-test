#include "vocab_embedding_r471.h"
#include <cuda_runtime_api.h>
#include <cstdint>
namespace {
__global__ void lookup(unsigned short*out,const unsigned short*weight,const int64_t*ids,std::size_t tokens,int rank){
 std::size_t z=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(z>=tokens*6144)return;
 int64_t global_id=ids[z/6144];
 // Original text-model CPU ID validation already enforces 0<=global_id<154880.
 // Exactly one real TP16 rank owns each row. Other ranks contribute BF16 +0.
 int64_t local=global_id-int64_t(rank)*9680;
 out[z]=(local>=0&&local<9680)?weight[std::size_t(local)*6144+z%6144]:0;
}
}
extern "C" int glm_vocab_embedding_r471(void*out,const void*weight,const void*ids,std::size_t tokens,int rank,void*stream){
 if(!out||!weight||!ids||!tokens||tokens>4096||rank<0||rank>=16)return 1;
 lookup<<<(tokens*6144+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<unsigned short*>(out),static_cast<const unsigned short*>(weight),static_cast<const int64_t*>(ids),tokens,rank);
 return int(cudaGetLastError());
}
