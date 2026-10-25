#include "provider_r9150.hpp"
#include "raw_r9150.hpp"
#include <atomic>
#include <array>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
namespace kda_math_r9150 {
#include "math_r9150.cuh"
}
#include <set>
namespace infini::ops::kda_threads_detail_r9150 {
std::array<std::array<std::atomic<uint64_t>,4>,8>counts{};
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<int64_t> Indices(const void* p,size_t n,bool i64,cudaStream_t stream){
  std::vector<int64_t> values(n);
  if(i64){Check(cudaMemcpyAsync(values.data(),p,n*8,cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));}
  else {std::vector<int32_t> small(n);Check(cudaMemcpyAsync(small.data(),p,n*4,cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));for(size_t i=0;i<n;++i)values[i]=small[i];}
  return values;
}
template<class T,class G> void Run(void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream,int threads,cudaEvent_t start,cudaEvent_t end){
  auto& s=l.strides;
  Check(cudaEventRecord(start,stream));
  Check(cudaMemcpyAsync(snapshot,state,l.pool*l.heads*l.dim*l.dim*sizeof(T),cudaMemcpyDeviceToDevice,stream));
  kda_math_r9150::kimiDeltaAttentionRecurrentCudaKernel<T,G><<<dim3(l.requests,l.heads,l.dim),threads,(3*l.dim+threads)*sizeof(float),stream>>>(
    static_cast<T*>(out),static_cast<T*>(snapshot),nullptr,static_cast<T*>(state),static_cast<const T*>(q),static_cast<const T*>(k),static_cast<const T*>(v),
    static_cast<const G*>(gate),static_cast<const G*>(beta),alog,bias,offsets,reads,writes,l.i64[0],l.i64[1],l.i64[2],l.norm,true,true,
    l.tokens,l.dim,l.pool,l.scale,l.lower,s[0][0],s[0][1],s[0][2],s[1][0],s[1][1],s[1][2],s[1][3],0,0,0,0,
    s[2][0],s[2][1],s[2][2],s[3][0],s[3][1],s[3][2],s[4][0],s[4][1],s[4][2],s[5][0],s[5][1],s[5][2],s[6][0],s[6][1],s[6][2],1,s[7][0]);
  Check(cudaGetLastError());
  Check(cudaEventRecord(end,stream));
}
template<class T> void Gate(DataType g,void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream,int threads,cudaEvent_t start,cudaEvent_t end){
  switch(g){case DataType::kFloat32:Run<T,float>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream,threads,start,end);break;
    case DataType::kFloat16:Run<T,half>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream,threads,start,end);break;
    case DataType::kBFloat16:Run<T,__nv_bfloat16>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream,threads,start,end);break;
    default:throw std::invalid_argument("KDA gate dtype unsupported");}
}
void Launch(DataType data,DataType g,void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream,int threads,cudaEvent_t start,cudaEvent_t end){
  auto off=Indices(offsets,l.requests+1,l.i64[0],stream),ri=Indices(reads,l.requests,l.i64[1],stream),wi=Indices(writes,l.requests,l.i64[2],stream);
  std::set<int64_t> unique;
  if(off.front()!=0||off.back()!=int64_t(l.tokens))throw std::invalid_argument("KDA token offsets mismatch");
  for(size_t i=0;i<l.requests;++i)
    if(off[i]>off[i+1]||ri[i]<0||uint64_t(ri[i])>=l.pool||wi[i]<0||uint64_t(wi[i])>=l.pool||!unique.insert(wi[i]).second)
      throw std::invalid_argument("KDA invalid indices or duplicate state writes");
  int device=-1;Check(cudaGetDevice(&device));if(device<0||device>=8|| (threads!=128&&threads!=256))throw std::invalid_argument("9150 device/threads");++counts[device][threads==128?1:0];counts[device][2]+=3;++counts[device][3];
  switch(data){case DataType::kFloat32:Gate<float>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream,threads,start,end);break;
    case DataType::kFloat16:Gate<half>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream,threads,start,end);break;
    case DataType::kBFloat16:Gate<__nv_bfloat16>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream,threads,start,end);break;
    default:throw std::invalid_argument("KDA data dtype unsupported");}
}
}

namespace {thread_local std::string error9150;}
extern "C" const char*modern_kimi_kda_threads_error_r9150(){return error9150.c_str();}
extern "C" int modern_kimi_kda_threads_counts_r9150(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=4)return 1;for(size_t j=0;j<4;++j)out[j]=infini::ops::kda_threads_detail_r9150::counts[device][j].load();return 0;}
extern "C" int modern_kimi_kda_threads_raw_r9150(int device,void*stream,void*workspace,size_t bytes,const infini::ops::Tensor*v,float scale,float lower,bool norm,int threads,void*start,void*end){try{
 using namespace infini::ops;int actual=-1;if(!v||!workspace||!start||!end||(threads!=128&&threads!=256)||cudaGetDevice(&actual)!=cudaSuccess||actual!=device)throw std::invalid_argument("9150 currentdevice/storage/threads");if(v[2].ndim()!=4||v[2].dtype()!=DataType::kBFloat16||v[2].shape()[2]!=6||v[2].shape()[3]!=128||v[2].shape()[1]>64||v[10].numel()>32)throw std::invalid_argument("9150 bounded H6D128BF16T64C32");ThreadsKdaR9150 op(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],scale,lower,norm);op.threads=threads;op.start=static_cast<cudaEvent_t>(start);op.end=static_cast<cudaEvent_t>(end);op.set_stream(stream);op.set_workspace(workspace);op.set_workspace_size_in_bytes(bytes);op(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],scale,lower,norm);return 0;
}catch(const std::exception&e){error9150=e.what();return 1;}}
