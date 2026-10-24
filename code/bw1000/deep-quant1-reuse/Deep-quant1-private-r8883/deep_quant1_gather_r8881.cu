#include <cuda_runtime_api.h>
#include <cstdint>
namespace {
__global__ void Gather(unsigned char*out,std::uint32_t*os,const unsigned char*in,const std::uint32_t*is,const std::int64_t*rows,std::size_t count){
 const std::size_t row=blockIdx.x;if(row>=count)return;const std::size_t src=static_cast<std::size_t>(rows[row]);
 for(unsigned c=threadIdx.x;c<4096;c+=blockDim.x)out[row*4096+c]=in[src*4096+c];
 if(threadIdx.x==0)os[row]=is[src];
}
}
extern "C" int deep_quant1_gather_r8881(void*out,void*os,const void*in,const void*is,const void*rows,std::size_t count,void*stream){
 if(!out||!os||!in||!is||!rows||!count||count>24576)return int(cudaErrorInvalidValue);
 Gather<<<unsigned(count),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<unsigned char*>(out),static_cast<std::uint32_t*>(os),static_cast<const unsigned char*>(in),static_cast<const std::uint32_t*>(is),static_cast<const std::int64_t*>(rows),count);
 return int(cudaGetLastError());
}
