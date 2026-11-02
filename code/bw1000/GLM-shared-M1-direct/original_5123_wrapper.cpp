#include <ATen/ATen.h>
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
extern "C" int glm_routed_gate_r19(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_routed_down_r19(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_pack5106(void*,const void*,const void*,const void*,int,int,void*);
extern "C" int glm_scale5108(void*,const void*,const void*,const void*,const void*,const void*,int,int,int,int,int,void*);
namespace {
thread_local std::string error;
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
 auto packed=wrap(x,{int64_t(gate?routes/8:routes),K},at::kByte,device);auto joined=at::empty({int64_t(routes),N},bf);Blas blas(static_cast<hipStream_t>(stream));
 for(size_t position=0;position<items.size();){int M=items[position].rows.size(),limit=std::min(16,std::max(1,2048/M));size_t end=position+1;while(end<items.size()&&end-position<size_t(limit)&&items[end].rows.size()>=size_t(std::max(1,M/2)))++end;int E=end-position;
  std::vector<int64_t>ex,order,padded,valid;for(size_t j=position;j<end;++j){auto&v=items[j].rows;ex.push_back(items[j].expert);for(int m=0;m<M;++m){padded.push_back(m<int(v.size())?v[m]:0);if(m<int(v.size())){valid.push_back((j-position)*M+m);order.push_back(v[m]);}}}position=end;
  auto expert_ids=meta(ex,{E},device),pad=meta(padded,{E,M},device),valid_ids=meta(valid,{int64_t(valid.size())},device),order_ids=meta(order,{int64_t(order.size())},device);auto flat=pad.reshape({-1});auto selected=gate?at::floor_divide(flat,8):flat;
  auto a=packed.index_select(0,selected).view(at::kFloat8_e4m3fn).to(at::kBFloat16).view({E,M,B,128}).permute({0,2,1,3}).contiguous().view({E*B,M,128});auto weight=at::empty({E*B,128,N},bf);ck(glm_pack5106(weight.data_ptr(),w,up,expert_ids.data_ptr(),E,gate?0:1,stream),"pack");auto partial=at::empty({E,B,M,N},fp);bmm(blas,a,weight,partial,E,B,M,N);ck(glm_scale5108(partial.data_ptr(),pad.data_ptr(),expert_ids.data_ptr(),xs,ws,us,E,M,offset,gate?0:1,256,stream),"scale");auto values=partial.sum(1).to(at::kBFloat16).view({E*M,N}).index_select(0,valid_ids);joined.index_copy_(0,order_ids,values);
 }
 if(gate){wrap(out,{int64_t(routes),256},at::kBFloat16,device).copy_(joined.slice(1,0,256));wrap(outup,{int64_t(routes),256},at::kBFloat16,device).copy_(joined.slice(1,256,512));}else wrap(out,{int64_t(routes),6144},at::kBFloat16,device).copy_(joined);
}
int policy(){const char*p=std::getenv("GLM_GROUPED_ROUTED_R5123");return!p?0:(!std::strcmp(p,"0")?0:(!std::strcmp(p,"1")?1:-1));}
}
extern "C" const char*glm_grouped_error_r5123(){return error.c_str();}
extern "C" int glm_grouped_counts_r5123(int d,uint64_t*out,size_t n){if(d<0||d>7||!out||n!=5)return-1;for(int i=0;i<5;++i)out[i]=counts[d][i];return 0;}
extern "C" int glm_grouped_gate_r5123(void*g,void*u,const void*x,const void*xs,const void*ids,const void*w,const void*ws,const void*up,const void*us,size_t routes,size_t offset,void*stream){
 int d=validate({g,u,x,xs,ids,w,ws,up,us},routes,offset);if(d<0)return d;counts[d][0]++;int mode=policy();if(mode<0)return-5;if(!mode||routes<4096){counts[d][3]++;return glm_routed_gate_r19(g,u,x,xs,ids,w,ws,up,us,routes,offset,stream);}try{run(g,u,x,xs,ids,w,ws,up,us,routes,offset,stream,true,d);counts[d][1]++;return 0;}catch(const std::exception&e){error=e.what();counts[d][4]++;return-4;}}
extern "C" int glm_grouped_down_r5123(void*out,const void*x,const void*xs,const void*ids,const void*w,const void*ws,size_t routes,size_t offset,void*stream){
 int d=validate({out,x,xs,ids,w,ws},routes,offset);if(d<0)return d;counts[d][0]++;int mode=policy();if(mode<0)return-5;if(!mode||routes<4096){counts[d][3]++;return glm_routed_down_r19(out,x,xs,ids,w,ws,routes,offset,stream);}try{run(out,nullptr,x,xs,ids,w,ws,nullptr,nullptr,routes,offset,stream,false,d);counts[d][2]++;return 0;}catch(const std::exception&e){error=e.what();counts[d][4]++;return-4;}}
