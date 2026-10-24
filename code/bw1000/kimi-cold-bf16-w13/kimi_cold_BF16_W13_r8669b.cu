// Bounded cold W13 dequant/BF16AB-F32C GEMM; not Model-qualified.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thread>
#include <string>
#include <stdexcept>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
#include "kimi_mxfp4_static_r326.cuh"
namespace infini::ops::mxfp4_detail {
__global__ void packRoutes4531(const int32_t*,int32_t*,int32_t*,size_t,size_t);
extern template __global__ void staticW2R326U<__nv_bfloat16>(__nv_bfloat16*,const __nv_bfloat16*,const int32_t*,const float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t,size_t);
// Exact active list from actual GPUcounts, not CPUfixture metadata.
__global__ void active8666(const int32_t*counts,int32_t*active,int32_t*count){
 if(threadIdx.x||blockIdx.x)return;int n=0;for(int e=0;e<896;++e)if(counts[e])active[n++]=e;*count=n;
}
__global__ void dequant8666(__nv_bfloat16*bank,const uint8_t*w,const uint8_t*scales,const int32_t*active,int total,int offset){
 constexpr size_t P=1792,S=112;size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size_t(32)*768*P)return;
 size_t slot=i/(768*P),row=(i/P)%768,p=i%P;float lo=0,hi=0;
 if(int(slot)+offset<total){size_t source=size_t(active[slot+offset])*768+row;mxfp4DecodePair(w[source*P+p],scales[source*S+p/16],lo,hi);}
 bank[i*2]=mxfp4Store<__nv_bfloat16>(lo);bank[i*2+1]=mxfp4Store<__nv_bfloat16>(hi);
}
__global__ void packX8666(__nv_bfloat16*x,const __nv_bfloat16*source,const int32_t*active,int total,int offset,const int32_t*counts,const int32_t*routes,int T){
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size_t(32)*16*3584)return;int slot=i/(16*3584),row=(i/3584)%16;size_t h=i%3584;__nv_bfloat16 value{};
 if(slot+offset<total){int e=active[slot+offset];if(row<counts[e]){int route=routes[size_t(e)*T*16+row];value=source[size_t(route/16)*3584+h];}}
 x[i]=value;
}
__global__ void activation_restore8666(__nv_bfloat16*out,const float*gemm,const int32_t*active,int total,int offset,const int32_t*counts,const int32_t*routes,int T){
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size_t(32)*16*384)return;int slot=i/(16*384),row=(i/384)%16;size_t h=i%384;
 if(slot+offset>=total)return;int e=active[slot+offset];if(row>=counts[e])return;int route=routes[size_t(e)*T*16+row];size_t base=(size_t(slot)*16+row)*768;
 out[size_t(route)*384+h]=mxfp4Store<__nv_bfloat16>(fusedMoeMxfp4Activate(gemm[base+h],gemm[base+384+h],size_t{2}));
}
}
namespace {thread_local std::string error8666;struct Context{int device;cudaStream_t stream;cublasHandle_t blas;std::thread::id owner;};void ck(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("vendorstatus="+std::to_string(int(s)));}}
extern "C" const char*kimi_bf16_cold_error_r8666(){return error8666.c_str();}
extern "C" void*kimi_bf16_cold_create_r8666(int device,void*stream,void*workspace,size_t bytes){Context*p=nullptr;try{
 int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=device||!workspace||bytes!=64*1024*1024)throw std::invalid_argument("owner/currentdevice/exactvendorworkspace64MiB");
 p=new Context{device,static_cast<cudaStream_t>(stream),nullptr,std::this_thread::get_id()};ck(cublasCreate(&p->blas));ck(cublasSetStream(p->blas,p->stream));ck(cublasSetPointerMode(p->blas,CUBLAS_POINTER_MODE_HOST));ck(cublasSetAtomicsMode(p->blas,CUBLAS_ATOMICS_NOT_ALLOWED));ck(cublasSetMathMode(p->blas,CUBLAS_DEFAULT_MATH));ck(cublasSetWorkspace(p->blas,workspace,bytes));return p;
}catch(const std::exception&e){error8666=e.what();if(p){if(p->blas)cublasDestroy(p->blas);delete p;}return nullptr;}}
extern "C" int kimi_bf16_cold_run_r8666(void*raw,size_t T,void*out,void*activation,const void*input,const void*ids,const void*routeweights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*active,void*count,void*bank,void*packedX,void*gemm){
 auto*c=static_cast<Context*>(raw);try{
 int current=-1;if(!c||c->owner!=std::this_thread::get_id()||cudaGetDevice(&current)!=cudaSuccess||current!=c->device||T!=16)throw std::invalid_argument("boundedT16/owner");
 using namespace infini::ops::mxfp4_detail;using B=__nv_bfloat16;auto st=c->stream;float one=1,zero=0;cudaError_t code=cudaMemsetAsync(counts,0,896*sizeof(int32_t),st);if(code!=cudaSuccess)throw std::runtime_error("counts");
 packRoutes4531<<<1,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),896,T);
 active8666<<<1,1,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(active),static_cast<int32_t*>(count));
 // This real4B device→host/sync and cold dequant of EVERY active chunk are
 // intentionally INCLUDED in completeprovider timings. No free CPU routeplan.
 int total=0;if(cudaMemcpyAsync(&total,count,4,cudaMemcpyDeviceToHost,st)!=cudaSuccess||cudaStreamSynchronize(st)!=cudaSuccess||total<1||total>256)throw std::runtime_error("actualactivecount");
 for(int offset=0;offset<total;offset+=32){
  dequant8666<<<(32*768*1792+255)/256,256,0,st>>>(static_cast<B*>(bank),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),static_cast<const int32_t*>(active),total,offset);
  packX8666<<<(32*16*3584+255)/256,256,0,st>>>(static_cast<B*>(packedX),static_cast<const B*>(input),static_cast<const int32_t*>(active),total,offset,static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),int(T));
  // Row-major X16x3584 @ W768x3584.T => row-major F32 C16x768.
  ck(cublasGemmStridedBatchedEx(c->blas,CUBLAS_OP_T,CUBLAS_OP_N,768,16,3584,&one,bank,CUDA_R_16BF,3584,768LL*3584,packedX,CUDA_R_16BF,3584,16LL*3584,&zero,gemm,CUDA_R_32F,768,16LL*768,32,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  activation_restore8666<<<(32*16*384+255)/256,256,0,st>>>(static_cast<B*>(activation),static_cast<const float*>(gemm),static_cast<const int32_t*>(active),total,offset,static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),int(T));
 }
 staticW2R326U<B><<<3584*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(routeweights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},size_t{896},size_t{3584},size_t{384},size_t{0});
 code=cudaGetLastError();if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));return 0;
}catch(const std::exception&e){error8666=e.what();return 1;}}
extern "C" int kimi_bf16_cold_destroy_r8666(void*raw){auto*c=static_cast<Context*>(raw);try{if(!c||c->owner!=std::this_thread::get_id())throw std::invalid_argument("destroyowner");if(cudaStreamSynchronize(c->stream)!=cudaSuccess)throw std::runtime_error("drain");ck(cublasDestroy(c->blas));delete c;return 0;}catch(const std::exception&e){error8666=e.what();return 1;}}
