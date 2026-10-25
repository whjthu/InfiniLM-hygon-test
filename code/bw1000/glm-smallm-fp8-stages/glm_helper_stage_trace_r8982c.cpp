#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <array>
#include <link.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
namespace {
using Launch=cudaError_t(*)(const void*,dim3,dim3,void**,size_t,cudaStream_t);
extern "C" { typedef cublasStatus_t (*Gemm)(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const void*,const void*,cudaDataType_t,int,long long,const void*,cudaDataType_t,int,long long,const void*,void*,cudaDataType_t,int,long long,int,cublasComputeType_t,cublasGemmAlgo_t); }
struct State {bool initialized=false,armed=false,in_blas=false;int next=0,error=0;size_t n=0,k=0,sc=0,ro=0,co=0;void*base=nullptr;std::array<std::array<uintptr_t,2>,8> text{};size_t ranges=0;cudaStream_t stream{};std::thread::id owner;std::array<cudaEvent_t,8> events{};std::array<uint64_t,4> hits{};};
thread_local State s;
Launch launch_real=nullptr;Gemm gemm_real=nullptr;
bool attempted=false,resolved=false;std::string resolution;
void* provider(const char*name,void*selfbase,bool fallback){
 dlerror();void*p=dlsym(RTLD_NEXT,name);const char*err=dlerror();std::string why=err?err:"";const char*route="RTLD_NEXT";
 if(!p&&fallback){void*h=dlopen("libcublas.so.12",RTLD_NOW|RTLD_NOLOAD);const char*loaderr=dlerror();if(h){dlerror();p=dlsym(h,name);const char*e=dlerror();if(e)why=e;}else if(loaderr)why=loaderr;route="NOLOAD_EXISTING_LOCAL";}
 Dl_info d{};if(p&&(!dladdr(p,&d)||d.dli_fbase==selfbase)){p=nullptr;why="self_or_unidentified_provider_rejected";}
 resolution+=std::string(name)+" route="+route+" resolved="+(p?"1":"0")+" provider="+(p&&d.dli_fname?d.dli_fname:"")+" dlerror="+why+"; ";return p;
}
void mark(int slot){if(cudaEventRecord(s.events[slot],s.stream)!=cudaSuccess)s.error=1;}
bool ours(void*caller){auto a=reinterpret_cast<uintptr_t>(caller);for(size_t i=0;i<s.ranges;++i)if(a>=s.text[i][0]&&a<s.text[i][1])return true;return false;}
int bind_text(dl_phdr_info*p,size_t,void*){if(reinterpret_cast<void*>(p->dlpi_addr)!=s.base)return 0;for(int i=0;i<p->dlpi_phnum;++i){auto&h=p->dlpi_phdr[i];if(h.p_type==PT_LOAD&&(h.p_flags&PF_X)&&s.ranges<s.text.size())s.text[s.ranges++]={p->dlpi_addr+h.p_vaddr,p->dlpi_addr+h.p_vaddr+h.p_memsz};}return 1;}
size_t arg(void**a,int n){return *static_cast<const size_t*>(a[n]);}
void bad(){s.error=1;}
}
// Host-only symbol precheck inside the GPU allocation; no CUDA/device API call.
extern "C" int glm_helper_trace_precheck_r8982(){
 if(attempted)return resolved?0:-1;attempted=true;Dl_info own{};
 if(!dladdr(reinterpret_cast<void*>(&glm_helper_trace_precheck_r8982),&own))return -2;
 launch_real=reinterpret_cast<Launch>(provider("cudaLaunchKernel",own.dli_fbase,false));
 gemm_real=reinterpret_cast<Gemm>(provider("cublasGemmStridedBatchedEx",own.dli_fbase,true));
 resolved=launch_real&&gemm_real;return resolved?0:-3;
}
extern "C" const char*glm_helper_trace_resolution_r8982(){return resolution.c_str();}
extern "C" int glm_helper_trace_init_r8982(void*run,void*stream){
 if(!resolved)return -5;if(s.initialized||!run)return -1;Dl_info d{};if(!dladdr(run,&d)||!d.dli_fbase)return -2;
 s.base=d.dli_fbase;s.ranges=0;dl_iterate_phdr(bind_text,nullptr);if(!s.ranges)return -4;s.stream=static_cast<cudaStream_t>(stream);s.owner=std::this_thread::get_id();
 for(auto&e:s.events)if(cudaEventCreate(&e)!=cudaSuccess){for(auto&v:s.events)if(v){cudaEventDestroy(v);v=nullptr;}return -3;}
 s.initialized=true;return 0;
}
extern "C" int glm_helper_trace_begin_r8982(int enabled,size_t n,size_t k,size_t sc,size_t ro,size_t co){
 if(!s.initialized||s.owner!=std::this_thread::get_id()||s.armed||(enabled!=0&&enabled!=1)||!n||!k||k%128||co%128||ro%128||co/128+k/128>sc)return -1;
 s.n=n;s.k=k;s.sc=sc;s.ro=ro;s.co=co;s.next=0;s.error=0;s.hits={};s.armed=bool(enabled);return 0;
}
// Caller has already drained the original ownerstream, for both arms.
extern "C" int glm_helper_trace_end_r8982(double*out,uint64_t*hits){
 if(!s.initialized||s.owner!=std::this_thread::get_id()||!out||!hits)return -1;bool was=s.armed;s.armed=false;
 for(int j=0;j<4;++j){hits[j]=s.hits[j];out[j]=0;if(was){float ms=0;if(cudaEventElapsedTime(&ms,s.events[2*j],s.events[2*j+1])!=cudaSuccess)bad();out[j]=ms;}}
 return s.error||(was&&(s.next!=4||s.hits!=std::array<uint64_t,4>{1,1,1,1}))?-2:0;
}
extern "C" int glm_helper_trace_close_r8982(){
 if(!s.initialized||s.armed||s.owner!=std::this_thread::get_id())return -1;
 // Original fixture has drained before this call; no new stream/lifetime policy.
 int rc=0;for(auto&e:s.events){if(cudaEventDestroy(e)!=cudaSuccess)rc=-2;e=nullptr;}s.initialized=false;return rc;
}
extern "C" cudaError_t cudaLaunchKernel(const void*f,dim3 grid,dim3 block,void**args,size_t shared,cudaStream_t stream){
 Launch real=launch_real;if(!real)return cudaErrorSharedObjectSymbolNotFound;
 const bool observe=s.armed&&!s.in_blas&&ours(__builtin_return_address(0));int j=-1;
 if(observe){j=s.next;if((j!=0&&j!=1&&j!=3)||stream!=s.stream||!args){bad();j=-1;}
  else if(j<2){if(arg(args,2)!=(j==0?8:s.n)||arg(args,3)!=s.k||arg(args,4)!=s.co/128||arg(args,5)!=s.co)bad();}
  else if(arg(args,4)!=8||arg(args,5)!=s.n||arg(args,6)!=s.sc||arg(args,7)!=s.ro||arg(args,8)!=s.co/128||arg(args,9)!=s.k/128||arg(args,10)!=0)bad();
  if(j>=0)mark(2*j);
 }
 auto rc=real(f,grid,block,args,shared,stream);
 if(observe&&j>=0){mark(2*j+1);++s.hits[j];++s.next;if(rc!=cudaSuccess)bad();}return rc;
}
extern "C" cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const void*alpha,const void*a,cudaDataType_t at,int lda,long long sa,const void*b,cudaDataType_t bt,int ldb,long long sb,const void*beta,void*c,cudaDataType_t ct,int ldc,long long cs,int batches,cublasComputeType_t compute,cublasGemmAlgo_t algo){
 Gemm real=gemm_real;if(!real)return CUBLAS_STATUS_NOT_INITIALIZED;
 const bool observe=s.armed&&ours(__builtin_return_address(0));
 if(observe){if(s.next!=2||m!=int(s.n)||n!=8||k!=128||batches!=int(s.k/128)||at!=CUDA_R_16BF||bt!=CUDA_R_16BF||ct!=CUDA_R_32F||compute!=CUBLAS_COMPUTE_32F||ta!=CUBLAS_OP_T||tb!=CUBLAS_OP_N||lda!=128||ldb!=128||ldc!=int(s.n)||sa!=static_cast<long long>(s.n*128)||sb!=8*128||cs!=static_cast<long long>(8*s.n))bad();mark(4);s.in_blas=true;}
 auto rc=real(h,ta,tb,m,n,k,alpha,a,at,lda,sa,b,bt,ldb,sb,beta,c,ct,ldc,cs,batches,compute,algo);
 if(observe){s.in_blas=false;mark(5);++s.hits[2];++s.next;if(rc!=CUBLAS_STATUS_SUCCESS)bad();}return rc;
}
