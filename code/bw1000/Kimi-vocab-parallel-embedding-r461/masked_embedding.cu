#include <cuda_runtime.h>
#include "embedding_index_r461.hpp"
#include <cstdint>
#include <cstddef>

namespace {
// No arithmetic conversion: copy BF16 payload bits, or contribute positive zero.
__global__ void masked_embedding(std::uint16_t *out,const std::int64_t *ids,
                                const std::uint16_t *weight,std::size_t rank) {
    const std::size_t column=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(column>=kimi_vocab_r461::hidden)return;
    const std::size_t token=blockIdx.y,i=token*kimi_vocab_r461::hidden+column;
    const auto id=ids[token];
    out[i]=kimi_vocab_r461::owns(id,rank)
        ?weight[kimi_vocab_r461::local_index(id,rank,column)]
        :std::uint16_t(0);
}
}
extern "C" int modern_kimi_masked_embedding_r461(int device,void *out,const void *ids,
        const void *weight,std::size_t tokens,std::size_t rank,void *stream) {
    if(device<0||device>=8||!out||!ids||!weight||!tokens||tokens>512||rank>=32)return int(cudaErrorInvalidValue);
    int current=-1;auto status=cudaGetDevice(&current);
    if(status!=cudaSuccess||current!=device)return int(cudaErrorInvalidDevice);
    const dim3 grid((kimi_vocab_r461::hidden+255)/256,tokens);
    masked_embedding<<<grid,256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<std::uint16_t*>(out),static_cast<const std::int64_t*>(ids),
        static_cast<const std::uint16_t*>(weight),rank);
    return int(cudaGetLastError());
}
