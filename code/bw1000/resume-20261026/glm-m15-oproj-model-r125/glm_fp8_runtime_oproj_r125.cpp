#include "glm_fp8_runtime_r809.hpp"
#include "glm_fp8_cache_m15_r119.hpp"
#include "glm_m15_oproj_geometry_r125.hpp"
#include "glm_batch_cache_geometry_r3068.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
namespace rt=infinilm::runtime;
namespace {
thread_local std::string error;
struct Owner {
 rt::Runtime*runtime{};int device{};void*stream{};void*handle{};bool batch_enabled{false};bool m15_enabled{false};void*m15_handle{};uint64_t m15_eligible{},m15_selected{},m15_off{},m15_rejected{},m15_errors{},m15_switches{};uint64_t batch32{},batch64{},batch_fallback{},batch_switches{};
 ~Owner(){if(m15_handle&&glm_fp8_cache_destroy_r119(m15_handle,runtime,stream))std::terminate();if(handle&&glm_fp8_cache_destroy_r807(handle,runtime,stream))std::terminate();}
};
struct Store {
 std::map<rt::Runtime*,std::unique_ptr<Owner>>owners;
 std::array<std::array<uint64_t,10>,8>released{};
 std::array<std::array<uint64_t,17>,8>released_m15{};
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
void glm_m15_oproj_policy_r125(rt::Runtime&r,bool enabled){
 r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("r125 policy recording");auto&c=owner(r);if(c.m15_enabled!=enabled){c.m15_enabled=enabled;++c.m15_switches;}
}
bool fp8_cached_r809(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t ro,size_t co){
 r.assertOwnerThread();size_t m=x->size(0);
 if(m>=32){
  auto it=store.owners.find(&r);const bool enabled=it!=store.owners.end()&&it->second->batch_enabled;
  const bool admitted=enabled&&x->ndim()==2&&w->ndim()==2&&ws->ndim()==2&&
    glm_batch_cache_r3068::fits(m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co);
  if(!admitted){if(enabled)++it->second->batch_fallback;return fp8_prefill_r780(r,out,x,xs,w,ws,ro,co);}
 }
 auto m15=store.owners.find(&r);const bool geometry15=m==15&&w->ndim()==2&&x->ndim()==2&&ws->ndim()==2&&glm_m15_oproj_r125::fits(m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co);
 bool select15=false;
 if(m==15&&m15!=store.owners.end()){
  auto&v=*m15->second;if(geometry15){++v.m15_eligible;select15=v.m15_enabled;if(!select15)++v.m15_off;}else ++v.m15_rejected;
 }
 if(m!=8&&m!=16&&m!=32&&m!=64&&!select15)return false;
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||out->dtype()!=rt::DataType::kBFloat16||x->dtype()!=rt::DataType::kUInt8||w->dtype()!=x->dtype()||xs->dtype()!=rt::DataType::kFloat32||ws->dtype()!=xs->dtype()||out->device()!=r.device()||x->device()!=r.device()||xs->device()!=r.device()||w->device()!=r.device()||ws->device()!=r.device()||!out->is_contiguous()||!x->is_contiguous()||!xs->is_contiguous()||!w->is_contiguous()||!ws->is_contiguous()||out->ndim()!=2||x->ndim()!=2||w->ndim()!=2||xs->ndim()!=2||ws->ndim()!=2||out->size(0)!=m||out->size(1)!=w->size(0)||x->size(1)!=w->size(1)||xs->size(0)!=m||xs->size(1)!=ws->size(1))throw std::invalid_argument("809 owned contiguous FP8 M8/16 geometry differs");
 auto&c=owner(r);
 if(select15){try{
  if(!c.m15_handle){c.m15_handle=glm_fp8_cache_create_r119(c.device,c.stream,&r);if(!c.m15_handle)throw std::runtime_error(glm_fp8_cache_error_r119());}
  if(glm_fp8_cache_run_r119(c.m15_handle,&r,r.stream(),1,1,m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_cache_error_r119());
  ++c.m15_selected;modern_glm_fp8_count_r780(r.device().index());return true;
 }catch(...){++c.m15_errors;throw;}}
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
extern "C" int modern_glm_m15_oproj_stats_r125(int device,uint64_t*out,size_t count){try{
 if(device<0||device>=8||!out||count!=17)throw std::invalid_argument("r125 stats bounds");const Owner*found=nullptr;
 for(const auto&[id,p]:store.owners)if(p->device==device){id->assertOwnerThread();if(found)throw std::logic_error("r125 ambiguous owner");found=p.get();}
 if(!found){std::copy(store.released_m15[device].begin(),store.released_m15[device].end(),out);return 0;}
 const auto&c=*found;const std::array<uint64_t,7>head{uint64_t(c.m15_enabled),c.m15_eligible,c.m15_selected,c.m15_off,c.m15_rejected,c.m15_errors,c.m15_switches};std::copy(head.begin(),head.end(),out);std::fill(out+7,out+17,0);
 if(c.m15_handle&&glm_fp8_cache_stats_r119(c.m15_handle,out+7,10))throw std::runtime_error(glm_fp8_cache_error_r119());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_glm_fp8_runtime_clear_r809(rt::Runtime*r){try{
 if(!r)throw std::invalid_argument("809 missing Runtime");r->assertOwnerThread();auto i=store.owners.find(r);if(i==store.owners.end())return 0;
 auto&p=*i->second;r->syncStream();
 if(p.m15_handle&&glm_fp8_cache_clear_r119(p.m15_handle,r,p.stream))throw std::runtime_error(glm_fp8_cache_error_r119());
 auto&released15=store.released_m15.at(p.device);if(modern_glm_m15_oproj_stats_r125(p.device,released15.data(),17))throw std::runtime_error(error);
 if(released15[7]!=released15[8]||released15[12]||released15[13])throw std::logic_error("r125 private cache not released");released15[0]=0;p.m15_enabled=false;
 if(glm_fp8_cache_clear_r807(p.handle,r,p.stream))throw std::runtime_error(glm_fp8_cache_error_r807());
 auto&counts=store.released.at(p.device);if(glm_fp8_cache_stats_r807(p.handle,counts.data(),counts.size()))throw std::runtime_error(glm_fp8_cache_error_r807());
 if(counts[0]!=counts[1]||counts[5]||counts[6])throw std::logic_error("809 plans not fully released");
 store.owners.erase(i);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
