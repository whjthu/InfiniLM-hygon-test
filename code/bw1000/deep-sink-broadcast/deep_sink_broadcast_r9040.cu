#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
__global__ void sink_r9040(uint32_t*out,const uint32_t*in,size_t R,size_t H,size_t N){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<R*H)out[i*(N+1)+N]=in[i%H];}
extern "C" int deep_sink_broadcast_r9040(void*out,const void*in,size_t R,size_t H,size_t N,void*stream){if(!out||!in||R<2||H!=8||!N)return 1;sink_r9040<<<(R*H+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint32_t*>(out),static_cast<const uint32_t*>(in),R,H,N);return int(cudaGetLastError());}
