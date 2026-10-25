#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <c10/core/InferenceMode.h>
#include <hip/hip_runtime.h>

#include <vector>
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <dlfcn.h>
extern "C" int grid_cuda_launch8998(const void*,dim3,dim3,void**,size_t,void*) __asm__("cudaLaunchKernel");
namespace grid8998 {
using Launch=int(*)(const void*,dim3,dim3,void**,size_t,void*);
Launch original=nullptr;
struct Device {std::atomic<unsigned>word{0};std::array<std::atomic<uint64_t>,5>counts{};};
std::array<Device,8>devices;
struct Frame;thread_local Frame*frame=nullptr;
struct Frame{int device;bool enabled;Frame*prior;explicit Frame(int d):device(d),enabled(false),prior(frame){if(prior)throw std::runtime_error("8998 nested down frame");enabled=devices[d].word.fetch_add(2,std::memory_order_acq_rel)&1;frame=this;devices[d].counts[0].fetch_add(1,std::memory_order_relaxed);}~Frame(){frame=prior;devices[device].word.fetch_sub(2,std::memory_order_release);}};
struct Scope;thread_local Scope*inside=nullptr;
struct Scope{Frame&owner;int E,M,launches=0;void*stream;Scope*prior;Scope(int e,int m,void*s):owner(*frame),E(e),M(m),stream(s),prior(inside){if(prior)throw std::runtime_error("8998 nested CERT scope");inside=this;devices[owner.device].counts[1].fetch_add(1,std::memory_order_relaxed);}~Scope(){inside=prior;}void finish(){if(launches!=2)throw std::runtime_error("8998 original CERT/FIX sequence");}};
}
extern "C" int glm_prefill_grid_resolve_r8998(){try{void*h=dlopen("libcudart.so.12",RTLD_NOW|RTLD_NOLOAD);if(!h)return 1;void*p=dlsym(h,"cudaLaunchKernel");if(!p||p==reinterpret_cast<void*>(&grid_cuda_launch8998))return 2;Dl_info d{};if(!dladdr(p,&d)||!d.dli_fname||std::string(d.dli_fname).find("libcudart.so")==std::string::npos)return 3;grid8998::original=reinterpret_cast<grid8998::Launch>(p);return 0;}catch(...){return 4;}}
extern "C" int glm_prefill_grid_policy_r8998(int device,int mode){if(device<0||device>7||mode<0||mode>1)return-1;int current=-1;if(hipGetDevice(&current)!=hipSuccess||current!=device)return-2;auto&s=grid8998::devices[device];unsigned expected=s.word.load(std::memory_order_acquire);for(;;){if(expected>=2)return-3;if(s.word.compare_exchange_weak(expected,unsigned(mode),std::memory_order_acq_rel))return 0;}}
extern "C" int glm_prefill_grid_counts_r8998(int device,uint64_t*out,size_t n){if(device<0||device>7||!out||n!=5)return-1;for(size_t j=0;j<5;++j)out[j]=grid8998::devices[device].counts[j].load(std::memory_order_relaxed);return 0;}
extern "C" int grid_cuda_launch8998(const void*function,dim3 grid,dim3 block,void**args,size_t shared,void*stream){using namespace grid8998;if(!original)return 999;Scope*s=inside;if(!s)return original(function,grid,block,args,shared,stream);auto&stats=devices[s->owner.device].counts;int j=s->launches;if(j>1||grid.x!=unsigned(j?256:24*s->E*s->M)||grid.y!=1||grid.z!=1||block.x!=256||block.y!=1||block.z!=1||shared||stream!=s->stream){stats[4].fetch_add(1,std::memory_order_relaxed);return 999;}bool changed=j==1&&s->owner.enabled;if(changed)grid.x=1024;int rc=original(function,grid,block,args,shared,stream);++s->launches;if(j==1)stats[2].fetch_add(1,std::memory_order_relaxed);if(changed)stats[3].fetch_add(1,std::memory_order_relaxed);if(rc)stats[4].fetch_add(1,std::memory_order_relaxed);return rc;}
extern "C" int glm_routed_gate_r19(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_routed_down_r19(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_pack5106(void*,const void*,const void*,const void*,int,int,void*);
extern "C" int glm_scale5108(void*,const void*,const void*,const void*,const void*,const void*,int,int,int,int,int,void*);
extern "C" int glm_certified_repair_r5248(void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,void*,void*,int,int,int,int,void*);
extern "C" int glm_rowreuse_r5206(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,int,int,int,int,int,void*);
extern "C" int glm_gpu_dedup_r5374(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,int,int,int,void*);
extern "C" int glm_dedup_summary_r5386(void*,const void*,int,int,void*);
namespace {
thread_local std::string error;
struct DecodeScope{bool on=false;void*diagnostic=nullptr;};thread_local std::array<DecodeScope,8>decode_scope;
std::array<std::array<std::atomic<uint64_t>,4>,8>decode_counts{};
int decode_policy(){const char*p=std::getenv("GLM_DECODE_ROW_DEDUP_R5386");return!p?0:(!std::strcmp(p,"0")?0:(!std::strcmp(p,"1")?1:-1));}

std::array<std::array<std::atomic<uint64_t>,5>,8> counts{};
void ck(int code,const char*what){if(code)throw std::runtime_error(std::string(what)+":"+std::to_string(code));}
struct Blas{Blas(hipStream_t){}};
at::Tensor wrap(const void*p,at::IntArrayRef shape,at::ScalarType type,int device){return at::from_blob(const_cast<void*>(p),shape,at::TensorOptions().device(at::kCUDA,device).dtype(type));}
at::Tensor meta(std::vector<int64_t>&v,at::IntArrayRef shape,int device){return at::from_blob(v.data(),shape,at::TensorOptions().dtype(at::kLong)).to(at::Device(at::kCUDA,device),at::kLong,false,true);}
int device_of(const void*p){hipPointerAttribute_t a{};if(!p||hipPointerGetAttributes(&a,p)!=hipSuccess)return-1;return a.type==hipMemoryTypeDevice?a.device:-1;}
int validate(std::initializer_list<const void*>pointers,size_t routes,size_t offset){if(!routes||routes%8||routes>32768||offset>1792||offset%256)return-1;int current=-1;if(hipGetDevice(&current)!=hipSuccess||current<0||current>7)return-2;for(auto p:pointers)if(device_of(p)!=current)return-3;return current;}
void bmm(Blas&,const at::Tensor&a,const at::Tensor&w,at::Tensor&p,int E,int B,int M,int N){p=at::bmm(a,w,at::kFloat).view({E,B,M,N});}

void run(void*out,void*outup,const void*x,const void*xs,const void*ids,const void*w,const void*ws,const void*up,const void*us,size_t routes,size_t offset,void*stream,bool gate,int device){
 c10::InferenceMode inference;c10::hip::HIPStreamGuard stream_guard(c10::hip::getStreamFromExternal(static_cast<hipStream_t>(stream),device));
 const int K=gate?6144:256,B=K/128,N=gate?512:6144;auto byte=at::TensorOptions().device(at::kCUDA,device).dtype(at::kByte);auto bf=byte.dtype(at::kBFloat16),fp=byte.dtype(at::kFloat);
 auto id=wrap(ids,{int64_t(routes)},at::kLong,device);auto host=id.to(at::kCPU);auto data=host.data_ptr<int64_t>();std::array<std::vector<int64_t>,256> groups;
 for(size_t i=0;i<routes;++i){if(data[i]<0||data[i]>=256)throw std::runtime_error("route_expert_ID");groups[data[i]].push_back(i);}
 struct Item{int64_t expert;std::vector<int64_t> rows;};std::vector<Item>items;
 // Match baseline Python insertion order: first occurrence of an expert.
 std::array<bool,256>seen{};for(size_t i=0;i<routes;++i){int64_t e=data[i];if(seen[e])continue;seen[e]=true;auto&v=groups[e];for(size_t j=0;j<v.size();j+=1024)items.push_back({e,std::vector<int64_t>(v.begin()+j,v.begin()+std::min(v.size(),j+1024))});}
 std::stable_sort(items.begin(),items.end(),[](const Item&a,const Item&b){return a.rows.size()>b.rows.size();});
 auto packed=wrap(x,{int64_t(gate?routes/8:routes),K},at::kByte,device);auto joined=gate?at::Tensor():at::empty({int64_t(routes),N},bf);Blas blas(static_cast<hipStream_t>(stream));
 for(size_t position=0;position<items.size();){int M=items[position].rows.size(),limit=std::min(16,std::max(1,2048/M));size_t end=position+1;while(end<items.size()&&end-position<size_t(limit)&&items[end].rows.size()>=size_t(std::max(1,M/2)))++end;int E=end-position;
  std::vector<int64_t>ex,order,padded,valid;for(size_t j=position;j<end;++j){auto&v=items[j].rows;ex.push_back(items[j].expert);for(int m=0;m<M;++m){padded.push_back(m<int(v.size())?v[m]:-1);if(m<int(v.size())){valid.push_back((j-position)*M+m);order.push_back(v[m]);}}}position=end;
  auto expert_ids=meta(ex,{E},device),pad=meta(padded,{E,M},device);if(gate){ck(glm_rowreuse_r5206(out,outup,x,xs,w,ws,up,us,pad.data_ptr(),expert_ids.data_ptr(),E,M,offset,1,8,stream),"exact_R8_gate");continue;}auto valid_ids=meta(valid,{int64_t(valid.size())},device),order_ids=meta(order,{int64_t(order.size())},device);auto flat=pad.reshape({-1});auto safe_flat=at::clamp_min(flat,0);auto selected=gate?at::floor_divide(safe_flat,8):safe_flat;
  auto a=packed.index_select(0,selected).view(at::kFloat8_e4m3fn).to(at::kBFloat16).view({E,M,B,128}).permute({0,2,1,3}).contiguous().view({E*B,M,128});auto weight=at::empty({E*B,128,N},bf);ck(glm_pack5106(weight.data_ptr(),w,up,expert_ids.data_ptr(),E,gate?0:1,stream),"pack");at::Tensor partial;bmm(blas,a,weight,partial,E,B,M,N);auto absolute=at::bmm(a.abs(),weight.abs(),at::kFloat).view({E,B,M,N});auto counter=at::zeros({1},byte.dtype(at::kInt));auto indices=at::empty({int64_t(E)*M*N},byte.dtype(at::kInt));auto values=at::empty({E,M,N},bf);grid8998::Scope repair_scope(E,M,stream);ck(glm_certified_repair_r5248(values.data_ptr(),partial.data_ptr(),absolute.data_ptr(),pad.data_ptr(),expert_ids.data_ptr(),xs,ws,us,x,w,up,counter.data_ptr(),indices.data_ptr(),E,M,offset,gate?1:0,stream),"certified_BF16_midpoint_and_original_tree_repair");repair_scope.finish();values=values.view({E*M,N}).index_select(0,valid_ids);joined.index_copy_(0,order_ids,values);
 }
 if(!gate)wrap(out,{int64_t(routes),6144},at::kBFloat16,device).copy_(joined);
}
void run_decode(void*out,void*outup,const void*x,const void*xs,const void*ids,const void*w,const void*ws,const void*up,const void*us,size_t routes,size_t offset,void*stream,bool gate,int device){
 if(!stream)throw std::invalid_argument("decode_null_ownerstream");hipStreamCaptureStatus capture{};if(hipStreamIsCapturing(static_cast<hipStream_t>(stream),&capture)!=hipSuccess||capture!=hipStreamCaptureStatusNone)throw std::invalid_argument("decode_graph_capture_unsupported");
 c10::InferenceMode inference;c10::hip::HIPStreamGuard guard(c10::hip::getStreamFromExternal(static_cast<hipStream_t>(stream),device));auto opts=at::TensorOptions().device(at::kCUDA,device);auto pad=at::empty({256,int64_t(routes)},opts.dtype(at::kLong)),groupcounts=at::empty({256},opts.dtype(at::kInt)),active=at::zeros({514},opts.dtype(at::kInt)),rep=at::empty({int64_t(routes)},opts.dtype(at::kLong));
 ck(glm_gpu_dedup_r5374(out,outup,x,xs,ids,w,ws,up,us,pad.data_ptr(),groupcounts.data_ptr(),active.data_ptr(),rep.data_ptr(),routes,offset,gate?1:0,stream),"decode_strict_row_identity_original_tree");
 // Optional caller-owned diagnostic tensor stays alive through ownerstream drain.
 // NULL (normal timing/default) means no extra GPU kernel or D2H.
 if(decode_scope[device].diagnostic)ck(glm_dedup_summary_r5386(decode_scope[device].diagnostic,rep.data_ptr(),routes,gate?1:0,stream),"decode_identity_diagnostic");
}
bool supported_bmm_mode(){auto& c=at::globalContext();return !c.allowTF32CuBLAS()&&c.float32MatmulPrecision()==at::Float32MatmulPrecision::HIGHEST&&c.allowFP16ReductionCuBLAS()==at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK&&c.allowBF16ReductionCuBLAS()==at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK&&!c.allowFP16AccumulationCuBLAS();}
int policy(){const char*p=std::getenv("GLM_GROUPED_ROUTED_R5123");return!p?0:(!std::strcmp(p,"0")?0:(!std::strcmp(p,"1")?1:-1));}
}
extern "C" const char*glm_grouped_error_r5123(){return error.c_str();}
extern "C" int glm_grouped_counts_r5123(int d,uint64_t*out,size_t n){if(d<0||d>7||!out||n!=5)return-1;for(int i=0;i<5;++i)out[i]=counts[d][i];return 0;}
extern "C" int glm_grouped_gate_r5123(void*g,void*u,const void*x,const void*xs,const void*ids,const void*w,const void*ws,const void*up,const void*us,size_t routes,size_t offset,void*stream){
 int d=validate({g,u,x,xs,ids,w,ws,up,us},routes,offset);if(d<0)return d;counts[d][0]++;int mode=policy();if(mode<0)return-5;if(mode&&routes==128&&decode_scope[d].on){if(decode_policy()!=1){error="decode_policy_not_explicit_ON";return-7;}try{run_decode(g,u,x,xs,ids,w,ws,up,us,routes,offset,stream,true,d);counts[d][1]++;decode_counts[d][0]++;return 0;}catch(const std::exception&e){error=e.what();counts[d][4]++;decode_counts[d][2]++;return-8;}}if(!mode||routes<4096){counts[d][3]++;return glm_routed_gate_r19(g,u,x,xs,ids,w,ws,up,us,routes,offset,stream);}if(!supported_bmm_mode()){error="unsupported_ATen_matmul_mode";return-6;}try{run(g,u,x,xs,ids,w,ws,up,us,routes,offset,stream,true,d);counts[d][1]++;return 0;}catch(const std::exception&e){error=e.what();counts[d][4]++;return-4;}}
extern "C" int glm_grouped_down_r5123(void*out,const void*x,const void*xs,const void*ids,const void*w,const void*ws,size_t routes,size_t offset,void*stream){
 int d=validate({out,x,xs,ids,w,ws},routes,offset);if(d<0)return d;counts[d][0]++;int mode=policy();if(mode<0)return-5;if(mode&&routes==128&&decode_scope[d].on){if(decode_policy()!=1){error="decode_policy_not_explicit_ON";return-7;}try{run_decode(out,nullptr,x,xs,ids,w,ws,nullptr,nullptr,routes,offset,stream,false,d);counts[d][2]++;decode_counts[d][1]++;return 0;}catch(const std::exception&e){error=e.what();counts[d][4]++;decode_counts[d][2]++;return-8;}}if(!mode||routes<4096){counts[d][3]++;return glm_routed_down_r19(out,x,xs,ids,w,ws,routes,offset,stream);}if(!supported_bmm_mode()){error="unsupported_ATen_matmul_mode";return-6;}try{grid8998::Frame grid_frame(d);run(out,nullptr,x,xs,ids,w,ws,nullptr,nullptr,routes,offset,stream,false,d);counts[d][2]++;return 0;}catch(const std::exception&e){error=e.what();counts[d][4]++;return-4;}}

// Scope is thread-local and reset by the caller in finally after each synchronous
// Model forward. It neither caches weight identity nor crosses layer/epoch calls.
extern "C" int glm_dedup_scope_r5386(int device,int enabled,void*diagnostic){
 if(device<0||device>7||enabled<0||enabled>1||(!enabled&&diagnostic))return-1;
 if(enabled){int current=-1;if(hipGetDevice(&current)!=hipSuccess||current!=device||decode_policy()!=1)return-2;if(diagnostic&&device_of(diagnostic)!=device)return-3;}
 decode_scope[device]={enabled==1,diagnostic};return 0;
}
extern "C" int glm_dedup_scope_get_r5386(int device){if(device<0||device>7)return-1;return decode_scope[device].on?1:0;}
extern "C" int glm_dedup_counts_r5386(int device,uint64_t*out,size_t n){if(device<0||device>7||!out||n!=4)return-1;for(int i=0;i<4;++i)out[i]=decode_counts[device][i];return 0;}
