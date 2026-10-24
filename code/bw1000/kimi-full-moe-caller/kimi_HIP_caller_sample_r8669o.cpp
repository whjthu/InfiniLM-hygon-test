#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <array>
#include <cstdlib>
// DTK CUDA compatibility public types specify the same native error/kind/stream
// C ABI used by these four HIP exports. No kernel/stream/device state changes.
namespace {
using J=nlohmann::json;using Clock=std::chrono::steady_clock;
uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
struct Sample{unsigned phase,api;uint64_t ordinal,native_ns;int depth;void*frames[24];};
struct State{bool enabled=false,inside=false;int rank=-1;unsigned phase=0;uint64_t calls[3][4]{},native[3][4]{},wrapper[3][4]{};size_t used=0,cap[3][4]{};std::array<Sample,1024>samples;};
thread_local State state;thread_local std::string serialized;
template<class F>F resolve(const char*n){auto f=reinterpret_cast<F>(dlsym(RTLD_NEXT,n));if(!f)std::_Exit(89);return f;}
template<class Call>cudaError_t invoke(unsigned api,Call call){
 auto&s=state;if(!s.enabled||s.inside)return call();s.inside=true;auto t=ns();const unsigned p=s.phase;auto ordinal=++s.calls[p][api];Sample*sample=nullptr;
 if((ordinal<=2||ordinal%17==0)&&s.cap[p][api]<128&&s.used<s.samples.size()){
  sample=&s.samples[s.used++];sample->phase=p;sample->api=api;sample->ordinal=ordinal;++s.cap[p][api];sample->depth=backtrace(sample->frames,24);
 }
 auto before=ns();auto code=call();auto after=ns();s.native[p][api]+=after-before;if(sample)sample->native_ns=after-before;s.wrapper[p][api]+=ns()-t;s.inside=false;return code;
}
}
extern "C" void kimi_caller_begin_r8669o(int rank,int phase){if(rank<0||rank>=8||(phase!=1&&phase!=2))std::_Exit(90);if(phase==1){state.enabled=false;state.rank=rank;state.phase=1;state.used=0;for(unsigned p=0;p<3;++p)for(unsigned a=0;a<4;++a){state.calls[p][a]=state.native[p][a]=state.wrapper[p][a]=state.cap[p][a]=0;}}else state.phase=2;state.enabled=true;}
extern "C" const char*kimi_caller_end_r8669o(){state.enabled=false;auto&s=state;J j={{"rank",s.rank},{"clock","sameownersteady_clock"},{"sample_period",17},{"sample_cap_per_api_phase",128},{"records",J::array()},{"totals",J::array()}};const char*names[]={"hipGetDevice","hipSetDevice","hipMemcpyAsync","hipStreamSynchronize"};
 for(unsigned p=1;p<3;++p)for(unsigned a=0;a<4;++a)j["totals"].push_back({{"phase",p==1?"fullMoEforward":"outercompletiondrain"},{"API",names[a]},{"calls",s.calls[p][a]},{"native_duration_sum_ns",s.native[p][a]},{"wrapper_including_sampling_sum_ns",s.wrapper[p][a]}});
 for(size_t i=0;i<s.used;++i){auto&v=s.samples[i];J frames=J::array();for(int f=0;f<v.depth;++f){Dl_info d{};if(dladdr(v.frames[f],&d)&&d.dli_fname)frames.push_back({{"DSO",d.dli_fname},{"offset",reinterpret_cast<uintptr_t>(v.frames[f])-reinterpret_cast<uintptr_t>(d.dli_fbase)},{"symbol",d.dli_sname?d.dli_sname:"UNKNOWN"}});else frames.push_back({{"symbol","UNKNOWN"}});}j["records"].push_back({{"phase",v.phase},{"API",names[v.api]},{"ordinal",v.ordinal},{"native_ns",v.native_ns},{"frames",frames}});}
 j["limits"]="ActualAPI counts and sparsecaller samples; sampledcaller counts not exactfamily counts; instrumentedtime not ordinaryperformance or additivecriticalpath. Rawfileoffsets preservedremote.";serialized=j.dump();return serialized.c_str();}
extern "C" cudaError_t hipGetDevice(int*p){static auto f=resolve<decltype(&cudaGetDevice)>("hipGetDevice");return invoke(0,[&]{return f(p);});}
extern "C" cudaError_t hipSetDevice(int n){static auto f=resolve<decltype(&cudaSetDevice)>("hipSetDevice");return invoke(1,[&]{return f(n);});}
extern "C" cudaError_t hipMemcpyAsync(void*d,const void*s,size_t n,cudaMemcpyKind k,cudaStream_t stream){static auto f=resolve<decltype(&cudaMemcpyAsync)>("hipMemcpyAsync");return invoke(2,[&]{return f(d,s,n,k,stream);});}
extern "C" cudaError_t hipStreamSynchronize(cudaStream_t stream){static auto f=resolve<decltype(&cudaStreamSynchronize)>("hipStreamSynchronize");return invoke(3,[&]{return f(stream);});}
