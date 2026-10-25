#include "glm_fp8_runtime_r809.hpp"
#include "glm_batch_cache_geometry_r3068.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <atomic>
#include <chrono>
#include <dlfcn.h>
#include <limits>
#include <infini/rt/device_guard.h>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <mutex>
namespace rt=infinilm::runtime;

// Existing2040 reuse, deliberately separate from807 geometry-only scratch cache.
namespace o_prepared_r8986 {
constexpr size_t cap=1280ull*1024*1024, plan_bytes=28344320;
std::array<std::atomic<int>,8> enabled{};
std::array<std::atomic<size_t>,8> reserve{};
std::array<std::mutex,8> device_locks;
// created,destroyed,hits,misses,disabled,geometry,memory,cap,runs,livebytes,
// highwater,min_observed_free,prepare_ns,source_rejections,clears,liveentries.
std::array<std::array<std::atomic<uint64_t>,16>,8> counts{};
struct Api {
 using Create=void*(*)(int,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,const void*,size_t,void*);
 using Run=int(*)(void*,int,void*,const void*,const void*,const void*,const void*,size_t);
 using Destroy=int(*)(void*);using Error=const char*(*)();using Mem=int(*)(size_t*,size_t*);
 void*library{};Create create{};Run run{};Destroy destroy{};Error error{};Mem mem{};
 Api(){
  library=dlopen("/refactor/candidates/glm-prepared-weight-r2040/libglm_prepared_r2040.so",RTLD_NOW|RTLD_LOCAL);
  if(!library)throw std::runtime_error(dlerror());
  create=reinterpret_cast<Create>(dlsym(library,"glm_fp8_prepared_create_r2040"));run=reinterpret_cast<Run>(dlsym(library,"glm_fp8_prepared_run_r2040"));destroy=reinterpret_cast<Destroy>(dlsym(library,"glm_fp8_destroy_r776"));error=reinterpret_cast<Error>(dlsym(library,"glm_fp8_error_r776"));
  if(!create||!run||!destroy||!error)throw std::runtime_error("8986 prepared ABI missing");
  Dl_info a{},b{},c{},d{};
  if(!dladdr(reinterpret_cast<void*>(create),&a)||!dladdr(reinterpret_cast<void*>(run),&b)||!dladdr(reinterpret_cast<void*>(destroy),&c)||!dladdr(reinterpret_cast<void*>(error),&d)||a.dli_fbase!=b.dli_fbase||a.dli_fbase!=c.dli_fbase||a.dli_fbase!=d.dli_fbase||std::string(a.dli_fname).find("glm-prepared-weight-r2040/libglm_prepared_r2040.so")==std::string::npos)throw std::runtime_error("8986 prepared ABI origin differs");
  void*cuda=dlopen("libcudart.so.12",RTLD_NOW|RTLD_NOLOAD);if(!cuda)throw std::runtime_error("8986 existing cudart missing");
  mem=reinterpret_cast<Mem>(dlsym(cuda,"cudaMemGetInfo"));if(!mem)throw std::runtime_error("8986 meminfo missing");
 }
};
Api&api(){static Api a;return a;}
using Key=std::array<uintptr_t,10>; // weight,dtype,M,N,K,scaleRows,scaleCols,rowOffset,colOffset,wsdtype
struct Entry{void*plan{};rt::Tensor weight;size_t generation{};};
struct Owner{std::weak_ptr<rt::Runtime>weak;void*stream{};int device{};size_t bytes{},generation{};std::map<Key,Entry>entries;};
struct Store{std::map<rt::Runtime*,Owner>owners;~Store(){if(!owners.empty())std::terminate();}};
thread_local Store store;
void owned(rt::Runtime&r,const Owner&o){r.assertOwnerThread();auto strong=o.weak.lock();if(strong.get()!=&r||o.device!=r.device().index()||o.stream!=r.stream())throw std::invalid_argument("8986 Runtime identity/device/stream differs");}
bool attempt(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t ro,size_t co){
 const int dev=r.device().index();if(dev<0||dev>=8)return false;auto&c=counts[dev];
 if(!enabled[dev].load()){++c[4];return false;}
 if(x->shape()!=rt::Shape{8,2048}||w->shape()!=rt::Shape{6144,2048}||out->shape()!=rt::Shape{8,6144}||xs->shape()!=rt::Shape{8,128}||ws->shape()!=rt::Shape{48,128}||ro||co%2048||co+2048>16384){++c[5];return false;}
 if(r.isGraphRecording())throw std::invalid_argument("8986 graph recording unsupported");
 infini::rt::DeviceGuard guard{r.device().type(),dev,true};std::lock_guard<std::mutex> lock(device_locks[dev]);auto it=store.owners.find(&r);
 if(it==store.owners.end()){Owner o;o.weak=r.shared_from_this();o.stream=r.stream();o.device=dev;it=store.owners.emplace(&r,std::move(o)).first;}
 auto&o=it->second;owned(r,o);Key key{reinterpret_cast<uintptr_t>(w->data()),1,8,6144,2048,48,128,ro,co,0};auto e=o.entries.find(key);
 if(e==o.entries.end()){
  ++c[3];if(o.bytes>cap-plan_bytes||c[9].load()>cap-plan_bytes){++c[7];return false;}
  size_t free=0,total=0;if(api().mem(&free,&total))throw std::runtime_error("8986 actual device meminfo failed");
  auto low=c[11].load();while((!low||free<low)&&!c[11].compare_exchange_weak(low,free)){}
  const size_t keep=reserve[dev].load();if(!keep||free<plan_bytes||free-plan_bytes<keep){++c[6];return false;}
  if(o.generation==std::numeric_limits<size_t>::max())throw std::overflow_error("8986 generation overflow");const size_t generation=++o.generation;
  const auto start=std::chrono::steady_clock::now();void*p=api().create(dev,8,8,6144,2048,48,128,ro,co,w->data(),generation,r.stream());if(!p)throw std::runtime_error(api().error());
  c[12]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  try{e=o.entries.emplace(key,Entry{p,w,generation}).first;}catch(...){if(api().destroy(p))std::terminate();throw;}
  o.bytes+=plan_bytes;++c[0];++c[15];c[9]+=plan_bytes;auto high=c[10].load();while(high<c[9]&&!c[10].compare_exchange_weak(high,c[9].load())){}
 }else ++c[2];
 if(e->second.weight->data()!=w->data()||!e->second.generation){++c[13];throw std::logic_error("8986 immutable weight identity differs");}
 if(api().run(e->second.plan,2,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data(),e->second.generation))throw std::runtime_error(api().error());
 ++c[8];return true;
}
void clear(rt::Runtime&r){auto it=store.owners.find(&r);if(it==store.owners.end())return;auto&o=it->second;owned(r,o);std::lock_guard<std::mutex> lock(device_locks[o.device]);r.syncStream();auto&c=counts[o.device];
 for(auto e=o.entries.begin();e!=o.entries.end();){if(api().destroy(e->second.plan))throw std::runtime_error(api().error());e=o.entries.erase(e);o.bytes-=plan_bytes;++c[1];--c[15];c[9]-=plan_bytes;}++c[14];store.owners.erase(it);
}
}
extern "C" int modern_glm_o_prepared_policy_r8986(int device,int enable,size_t original_reserve_bytes){
 if(device<0||device>=8||(enable!=0&&enable!=1)||(enable&&!original_reserve_bytes))return 1;
 o_prepared_r8986::reserve[device]=original_reserve_bytes;o_prepared_r8986::enabled[device]=enable;return 0;
}
extern "C" int modern_glm_o_prepared_stats_r8986(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=16)return 1;for(size_t j=0;j<n;++j)out[j]=o_prepared_r8986::counts[device][j].load();return 0;}
namespace {
thread_local std::string error;
struct Owner {
 rt::Runtime*runtime{};int device{};void*stream{};void*handle{};bool batch_enabled{false};uint64_t batch32{},batch64{},batch_fallback{},batch_switches{};
 ~Owner(){if(handle&&glm_fp8_cache_destroy_r807(handle,runtime,stream))std::terminate();}
};
struct Store {
 std::map<rt::Runtime*,std::unique_ptr<Owner>>owners;
 std::array<std::array<uint64_t,10>,8>released{};
 ~Store(){if(!owners.empty())std::terminate();}
};
thread_local Store store;
Owner&owner(rt::Runtime&r){
 auto i=store.owners.find(&r);
 if(i==store.owners.end()){
  auto p=std::make_unique<Owner>();p->runtime=&r;p->device=r.device().index();p->stream=r.stream();p->handle=glm_fp8_cache_create_r807(p->device,p->stream,&r);
  if(!p->handle)throw std::runtime_error(glm_fp8_cache_error_r807());
  i=store.owners.emplace(&r,std::move(p)).first;
 }
 if(i->second->stream!=r.stream()||i->second->device!=r.device().index())throw std::invalid_argument("809 Runtime stream/device changed");
 return*i->second;
}
}
namespace infinilm::models::glm_moe_dsa {
void glm_fp8_batch_policy_r3068(rt::Runtime&r,bool enabled){
 r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("3068 policy during graph");
 auto&c=owner(r);c.batch_enabled=enabled;++c.batch_switches;
}
void glm_fp8_batch_stats_r3068(rt::Runtime&r,uint64_t*out,size_t count){
 r.assertOwnerThread();if(!out||count!=5)throw std::invalid_argument("3068 stats geometry");
 auto i=store.owners.find(&r);if(i==store.owners.end()){std::fill(out,out+5,0);return;}
 auto&c=*i->second;if(c.stream!=r.stream()||c.device!=r.device().index())throw std::logic_error("3068 stats changed owner");
 const std::array<uint64_t,5>v{uint64_t(c.batch_enabled),c.batch32,c.batch64,c.batch_fallback,c.batch_switches};std::copy(v.begin(),v.end(),out);
}
bool fp8_cached_r809(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t ro,size_t co){
 r.assertOwnerThread();size_t m=x->size(0);
 if(m>=32){
  auto it=store.owners.find(&r);const bool enabled=it!=store.owners.end()&&it->second->batch_enabled;
  const bool admitted=enabled&&x->ndim()==2&&w->ndim()==2&&ws->ndim()==2&&
    glm_batch_cache_r3068::fits(m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co);
  if(!admitted){if(enabled)++it->second->batch_fallback;return fp8_prefill_r780(r,out,x,xs,w,ws,ro,co);}
 }
 if(m!=8&&m!=16&&m!=32&&m!=64)return false;
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||out->dtype()!=rt::DataType::kBFloat16||x->dtype()!=rt::DataType::kUInt8||w->dtype()!=x->dtype()||xs->dtype()!=rt::DataType::kFloat32||ws->dtype()!=xs->dtype()||out->device()!=r.device()||x->device()!=r.device()||xs->device()!=r.device()||w->device()!=r.device()||ws->device()!=r.device()||!out->is_contiguous()||!x->is_contiguous()||!xs->is_contiguous()||!w->is_contiguous()||!ws->is_contiguous()||out->ndim()!=2||x->ndim()!=2||w->ndim()!=2||xs->ndim()!=2||ws->ndim()!=2||out->size(0)!=m||out->size(1)!=w->size(0)||x->size(1)!=w->size(1)||xs->size(0)!=m||xs->size(1)!=ws->size(1))throw std::invalid_argument("809 owned contiguous FP8 M8/16 geometry differs");
 if(o_prepared_r8986::attempt(r,out,x,xs,w,ws,ro,co)){modern_glm_fp8_count_r780(r.device().index());return true;}
 auto&c=owner(r);
 if(glm_fp8_cache_run_r807(c.handle,&r,r.stream(),1,1,m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_cache_error_r807());
 if(m==32)++c.batch32;else if(m==64)++c.batch64;
 modern_glm_fp8_count_r780(r.device().index());return true;
}
}
extern "C" const char*modern_glm_fp8_runtime_error_r809(){return error.c_str();}
extern "C" int modern_glm_fp8_runtime_stats_r809(int device,uint64_t*out,size_t count){try{
 if(device<0||device>=8||!out||count!=10)throw std::invalid_argument("809 stats bounds");
 const Owner*found=nullptr;
 for(const auto&[identity,p]:store.owners)if(p->device==device){identity->assertOwnerThread();if(found)throw std::logic_error("809 ambiguous Runtime owner");found=p.get();}
 if(found){if(glm_fp8_cache_stats_r807(found->handle,out,count))throw std::runtime_error(glm_fp8_cache_error_r807());}
 else std::copy(store.released[device].begin(),store.released[device].end(),out);
 return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_glm_fp8_runtime_clear_r809(rt::Runtime*r){try{
 if(!r)throw std::invalid_argument("809 missing Runtime");r->assertOwnerThread();o_prepared_r8986::clear(*r);auto i=store.owners.find(r);if(i==store.owners.end())return 0;
 auto&p=*i->second;r->syncStream();if(glm_fp8_cache_clear_r807(p.handle,r,p.stream))throw std::runtime_error(glm_fp8_cache_error_r807());
 auto&counts=store.released.at(p.device);if(glm_fp8_cache_stats_r807(p.handle,counts.data(),counts.size()))throw std::runtime_error(glm_fp8_cache_error_r807());
 if(counts[0]!=counts[1]||counts[5]||counts[6])throw std::logic_error("809 plans not fully released");
 store.owners.erase(i);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
