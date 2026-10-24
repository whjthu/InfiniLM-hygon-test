#include <cuda_runtime.h>
#include <cstdint>
namespace {
__global__ void mask_ep2_r8669zj(int32_t*out,const int32_t*ids,int begin){
 unsigned i=threadIdx.x;if(i<256){int e=ids[i];out[i]=(e>=begin&&e<begin+448)?e-begin:-1;}
}
}
extern "C" int modern_kimi_EP2_mask_r8669zj(int device,void*out,const void*ids,int begin,void*stream){
 int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=device||!out||!ids||(begin!=0&&begin!=448))return int(cudaErrorInvalidValue);
 mask_ep2_r8669zj<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<int32_t*>(out),static_cast<const int32_t*>(ids),begin);
 return int(cudaGetLastError());
}
