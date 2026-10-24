// Host launch boundaries ONLY. All __global__ functions are declarations;
// definitions and GPU code come from immutable current I192/packed objects.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace infini::ops::kimi_i192_r8669y {
__global__ void packRoutesI192R8669y(const int32_t*,int32_t*,int32_t*,size_t,size_t);
template<int TT>__global__ void buildJobsI192R8669y(const int32_t*,int32_t*,int32_t*,int32_t*,size_t);
template<int TT,int Tile>__global__ void groupedW13I192R8669y(__nv_bfloat16*,const __nv_bfloat16*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const uint8_t*,const uint8_t*,size_t);
template<typename T>__global__ void staticW2I192R8669y(T*,const T*,const int32_t*,const float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t,size_t);
}
namespace infini::ops::mxfp4_detail {
__global__ void packRoutes4531(const int32_t*,int32_t*,int32_t*,size_t,size_t);
template<int TT>__global__ void buildJobs4531(const int32_t*,int32_t*,int32_t*,int32_t*,size_t);
template<int TT,int Tile>__global__ void groupedW134531(__nv_bfloat16*,const __nv_bfloat16*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const uint8_t*,const uint8_t*,size_t);
template<typename T>__global__ void staticW2R326U(T*,const T*,const int32_t*,const float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t,size_t);
}
extern "C" void modern_kimi_provider_mark_r8669zn(void*,const char*);
extern "C" int modern_kimi_provider_phases_r8669zn(void*runtime,int kind,int device,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*counts,void*routes,void*jobExpert,void*jobStart,void*jobs,size_t E,size_t T,void*stream){
 using B=__nv_bfloat16;int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=device||T!=16||E!=(kind?448:896)||!runtime||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||!counts||!routes||!jobExpert||!jobStart||!jobs||(kind!=0&&kind!=1))return int(cudaErrorInvalidValue);auto st=static_cast<cudaStream_t>(stream);auto mark=[&](const char*n){modern_kimi_provider_mark_r8669zn(runtime,n);};
 mark("provider.pack.begin");auto code=cudaMemsetAsync(counts,0,E*sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);code=cudaMemsetAsync(jobs,0,sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);
 if(kind)infini::ops::mxfp4_detail::packRoutes4531<<<(T*16+255)/256,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),E,T);
 else infini::ops::kimi_i192_r8669y::packRoutesI192R8669y<<<(T*16+255)/256,256,0,st>>>(static_cast<const int32_t*>(ids),static_cast<int32_t*>(counts),static_cast<int32_t*>(routes),E,T);
 mark("provider.pack.end");
 if(kind)infini::ops::mxfp4_detail::buildJobs4531<2><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
 else infini::ops::kimi_i192_r8669y::buildJobsI192R8669y<2><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
 mark("provider.jobs.end");
 if(kind)infini::ops::mxfp4_detail::groupedW134531<2,4><<<640,256,2*2*4*256*sizeof(float),st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);
 else infini::ops::kimi_i192_r8669y::groupedW13I192R8669y<2,4><<<640,256,2*2*4*256*sizeof(float),st>>>(static_cast<B*>(activation),static_cast<const B*>(input),static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),T);
 code=cudaGetLastError();if(code!=cudaSuccess)return int(code);mark("provider.W13_fused_activation.end");
 if(kind)infini::ops::mxfp4_detail::staticW2R326U<B><<<3584*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},E,size_t{3584},size_t{384},size_t{0});
 else infini::ops::kimi_i192_r8669y::staticW2I192R8669y<B><<<3584*T,256,0,st>>>(static_cast<B*>(out),static_cast<const B*>(activation),static_cast<const int32_t*>(ids),static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T,size_t{16},E,size_t{3584},size_t{192},size_t{0});
 code=cudaGetLastError();if(code!=cudaSuccess)return int(code);mark("provider.W2_fused_scatter.end");return 0;
}
