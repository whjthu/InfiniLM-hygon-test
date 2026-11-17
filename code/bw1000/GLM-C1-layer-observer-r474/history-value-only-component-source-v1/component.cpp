// Source-only TP1 causal component. No entry may run before a separate GPU claim.
// One HOST binding against new179 B: its original expand_cached is unchanged;
// both arms share this actual module/weights/Runtime, avoiding dual-DSO ODR mixing.
#include "glm_moe_dsa_attention_prepare.hpp"
#include "history_value_only_v1.hpp"
#include "glm_rope_bound_r4279.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace rt=infinilm::runtime;
namespace model=infinilm::models::glm_moe_dsa;
namespace dist=infinilm::engine::distributed;
using J=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace {
constexpr std::size_t capacity=32768,floor_bytes=std::size_t{8}<<30;
thread_local std::string error,result;
void need(bool ok,const char*code){if(!ok)throw std::runtime_error(code);}
std::size_t floor(){std::size_t free=0,total=0;need(cudaMemGetInfo(&free,&total)==cudaSuccess&&free>=floor_bytes,"8GiB actual free floor");return free;}
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Held {
 rt::Runtime&r;std::vector<rt::Tensor>values;
 ~Held()noexcept{try{r.syncStream();}catch(...){std::terminate();}}
};
struct Context {
 std::unique_ptr<dist::CommunicationGroup>group;
 std::shared_ptr<rt::Runtime>runtime;
 std::unique_ptr<model::AttentionPrepare>prepare;
 std::set<std::string>loaded;
 bool ready=false,failed=false;
 rt::Runtime&owner(){need(runtime&&prepare,"component owner absent");runtime->assertOwnerThread();need(!runtime->isGraphRecording(),"graph unsupported");int d=-1;need(cudaGetDevice(&d)==cudaSuccess&&d==runtime->device().index(),"owner device changed");return *runtime;}
 void close(){
  if(!runtime){group.reset();return;}
  runtime->assertOwnerThread();runtime->syncStream();
  need(modern_glm_fp8_runtime_clear_r809(runtime.get())==0,"original FP8 Runtime clear");
  prepare.reset();runtime->trimMemory();std::weak_ptr<rt::Runtime>weak=runtime;runtime.reset();
  need(weak.expired(),"Runtime retained after normal release");group.reset();
 }
 ~Context()noexcept{try{close();}catch(...){std::terminate();}}
};
rt::DataType dtype(int code){if(code==0)return rt::DataType::kFloat32;if(code==1)return rt::DataType::kBFloat16;if(code==2)return rt::DataType::kUInt8;throw std::invalid_argument("dtype code");}
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
void finite(const std::uint16_t*p,std::size_t n){need(p&&n,"BF16 pointer/length");for(std::size_t i=0;i<n;++i)need((p[i]&0x7f80)!=0x7f80,"nonfinite BF16 input/output");}
}
extern "C" const char*glm_history_component_error(){return error.c_str();}
extern "C" const char*glm_history_component_result(){return result.c_str();}
extern "C" void*glm_history_component_create(const char*raw){
 try{
  need(raw,"configuration absent");J cfg=J::parse(raw);auto ctx=std::make_unique<Context>();
  // Actual group factory supplies RankInfo; never write its fields.
  dist::DistConfig topology(std::vector<int>{0});ctx->group=std::make_unique<dist::CommunicationGroup>(topology,rt::Device::Type::kHygon);
  need(ctx->group->tp_size()==1&&ctx->group->get_dist_config().tp_device_ids==std::vector<int>{0},"real group device topology");
  const auto info=ctx->group->get_rank_info(0);
  need(info.tp_size==1&&info.tp_rank==0&&info.pp_size==1&&info.pp_stage==0&&info.world_size==1&&info.world_rank==0&&info.device.index()==0,"actual TP1 rank geometry");
  infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};floor();
  ctx->runtime=rt::Runtime::create(info.device);ctx->runtime->enableOwnerDrainFailStop();auto&r=*ctx->runtime;
  model::rope_bound_r4279::Scope rope(r,true,capacity);
  ctx->prepare=std::make_unique<model::AttentionPrepare>(r,cfg,info);
  need(ctx->prepare->local_heads()==64,"TP1 original64 heads required");r.syncStream();floor();
  result=J{{"TP",1},{"heads",64},{"projection_N",28672},{"projection_K",512},{"capacity",capacity},{"Model_qualification",false}}.dump();return ctx.release();
 }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" const char*glm_history_component_keys(void*raw){
 try{need(raw,"keys absent context");auto&c=*static_cast<Context*>(raw);c.owner();J rows=J::array();auto state=c.prepare->state_dict();
  for(const auto&name:c.prepare->state_required_keys()){auto t=state.at(name).tensor();rows.push_back({{"name",name},{"shape",t->shape()},{"bytes",t->nbytes()},{"dtype_code",t->dtype()==rt::DataType::kFloat32?0:t->dtype()==rt::DataType::kBFloat16?1:t->dtype()==rt::DataType::kUInt8?2:-1}});}
  need(rows.size()==10,"all10 original module parameters");result=rows.dump();return result.c_str();
 }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" int glm_history_component_load(void*raw,const char*name,const void*bytes,std::size_t size,int code){
 auto*c=static_cast<Context*>(raw);try{need(c&&name&&bytes&&!c->ready&&!c->failed,"load state");auto&r=c->owner();need(!c->loaded.count(name),"duplicate parameter");
  auto t=c->prepare->state_dict().at(name).tensor();need(t&&t->dtype()==dtype(code)&&t->nbytes()==size&&t->is_contiguous(),"checkpoint metadata differs");
  Held held{r};auto source=rt::Tensor::from_blob(const_cast<void*>(bytes),t->shape(),dtype(code),{rt::Device::Type::kCpu,0});held.values.push_back(source);
  c->prepare->load_parameter(r,name,source);r.syncStream();c->loaded.insert(name);floor();return 0;
 }catch(const std::exception&e){if(c)c->failed=true;error=e.what();return 1;}
}
extern "C" int glm_history_component_finish(void*raw){
 auto*c=static_cast<Context*>(raw);try{need(c&&!c->ready&&!c->failed,"finish state");auto&r=c->owner();const auto keys=c->prepare->state_required_keys();need(keys.size()==10&&std::set<std::string>(keys.begin(),keys.end())==c->loaded,"required checkpoint keys not loaded");post(r,*c->prepare);c->ready=true;floor();return 0;
 }catch(const std::exception&e){if(c)c->failed=true;error=e.what();return 1;}
}
extern "C" int glm_history_component_pair(void*raw,const std::uint16_t*latent,std::size_t latent_bytes,const std::uint16_t*rotary,std::size_t rotary_bytes,const std::int64_t*ids,std::size_t rows,int candidate_first){
 auto*c=static_cast<Context*>(raw);try{
  need(c&&c->ready&&!c->failed&&latent&&rotary&&ids&&latent_bytes==capacity*512*2&&rotary_bytes==capacity*64*2&&rows>0&&rows<=8192&&(candidate_first==0||candidate_first==1),"controlled input geometry");
  for(std::size_t i=0;i<rows;++i)need(ids[i]==static_cast<std::int64_t>(i),"complete legal prefix IDs required");finite(latent,capacity*512);finite(rotary,capacity*64);
  auto&r=c->owner();floor();Held inputs{r};auto l=rt::Tensor::empty(r,{capacity,512},rt::DataType::kBFloat16),k=rt::Tensor::empty(r,{capacity,64},rt::DataType::kBFloat16),index=rt::Tensor::empty(r,{rows},rt::DataType::kInt64);inputs.values={l,k,index};
  r.memcpyH2D(l->data(),latent,latent_bytes,false);r.memcpyH2D(k->data(),rotary,rotary_bytes,false);r.memcpyH2D(index->data(),ids,rows*8,false);r.syncStream();floor();
  std::vector<std::uint16_t>a(rows*64*256),b(a.size());J calls=J::array();
  for(int order=0;order<2;++order){
   const bool candidate=order==0?candidate_first!=0:candidate_first==0;const auto started=Clock::now();double native_ms=0;std::uint64_t expands=0;
   {
    Held held{r};const auto before=model::v_on_demand_r400::completed_expand_calls;
    auto v=candidate?model::history_value_only_v1::selected_values(*c->prepare,r,l,k,index,held.values):model::v_on_demand_r400::selected_values(*c->prepare,r,l,k,index,held.values);
    r.syncStream();native_ms=ms(started);expands=model::v_on_demand_r400::completed_expand_calls-before;
    need(v&&v->shape()==rt::Shape{rows,64,256}&&v->dtype()==rt::DataType::kBFloat16&&v->is_contiguous()&&v->nbytes()==a.size()*2&&expands==1,"V output/counter contract");
    auto&out=candidate?b:a;r.memcpyD2H(out.data(),v->data(),v->nbytes());r.syncStream();finite(out.data(),out.size());
   }
   calls.push_back({{"candidate",candidate},{"native_gather_expand_sync_ms",native_ms},{"caller_readback_release_ms",ms(started)},{"completed_expand_delta",expands},{"free_bytes_after_release",floor()}});
  }
  std::size_t different=0,nonzero=0;for(std::size_t i=0;i<a.size();++i){different+=a[i]!=b[i];nonzero+=(a[i]&0x7fff)!=0;}
  // Readback full synthetic state after both arms, not an alleged historical cache.
  std::vector<std::uint16_t>checked(capacity*512);r.memcpyD2H(checked.data(),l->data(),latent_bytes);r.syncStream();need(std::memcmp(checked.data(),latent,latent_bytes)==0,"latent input changed");checked.resize(capacity*64);r.memcpyD2H(checked.data(),k->data(),rotary_bytes);r.syncStream();need(std::memcmp(checked.data(),rotary,rotary_bytes)==0,"rotary input changed");
  result=J{{"scope","TP1 controlled-input component; not TP16 Model"},{"TP",1},{"heads",64},{"rows",rows},{"capacity",capacity},{"V_elements",a.size()},{"bit_differences",different},{"nonfinite",0},{"nonzero",nonzero},{"input_unchanged",true},{"calls",calls},{"Model_GO",false}}.dump();need(different==0&&nonzero>0,"paired V bytes differ or vacuous zero output");return 0;
 }catch(const std::exception&e){if(c)c->failed=true;error=e.what();return 1;}
}
extern "C" int glm_history_component_guards(void*raw){
 auto*c=static_cast<Context*>(raw);try{need(c&&c->ready&&!c->failed,"guard state");auto&r=c->owner();Held held{r};std::size_t rejected=0;
  auto alloc=[&](rt::Shape s,rt::DataType d){auto t=rt::Tensor::empty(r,s,d);held.values.push_back(t);return t;};
  auto l=alloc({2,512},rt::DataType::kBFloat16),k=alloc({2,1,64},rt::DataType::kBFloat16);
  auto badwidth=alloc({2,511},rt::DataType::kBFloat16),badtype=alloc({2,512},rt::DataType::kUInt8),strided=alloc({2,1024},rt::DataType::kBFloat16)->narrow({{1,0,512}}),badrotary=alloc({2,64},rt::DataType::kBFloat16);
  need(!strided->is_contiguous(),"noncontiguous guard fixture");
  for(int candidate=0;candidate<2;++candidate){
   for(const auto&pair:std::vector<std::pair<rt::Tensor,rt::Tensor>>{{badwidth,k},{badtype,k},{strided,k},{l,badrotary}}){
    bool caught=false;try{if(candidate)c->prepare->expand_cached_values(r,pair.first,pair.second);else c->prepare->expand_cached(r,pair.first,pair.second);}catch(const std::invalid_argument&){caught=true;}
    need(caught,"original input guard unexpectedly accepted");++rejected;
   }
  }
  // These helper guards reject before a gather reads the uninitialised ID storage.
  // They are rejection fixtures, not fabricated history or model activations.
  for(std::size_t u:{std::size_t{8193},std::size_t{32768}}){auto ids=alloc({u},rt::DataType::kInt64);bool caught=false;
   try{model::history_value_only_v1::selected_values(*c->prepare,r,l,badrotary,ids,held.values);}catch(const std::invalid_argument&){caught=true;}
   need(caught,"single-block candidate boundary accepted");++rejected;
  }
  r.syncStream();result=J{{"planned_rejections",10},{"actual_rejections",rejected},{"graph_wrongowner_guards","not exercised; separate exact Runtime API qualification needed"},{"NoV_TP16_gate_coverage",false},{"free_bytes",floor()}}.dump();return 0;
 }catch(const std::exception&e){if(c)c->failed=true;error=e.what();return 1;}
}
extern "C" int glm_history_component_close(void*raw){
 try{need(raw,"close absent");auto*c=static_cast<Context*>(raw);c->close();delete c;result=J{{"normal_close",true},{"Runtime_weak_expired",true},{"free_bytes_after_close",floor()},{"global_live_buffer_counter",nullptr},{"global_live_buffer_counter_status","API/body not yet identified; not asserted zero"}}.dump();return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
