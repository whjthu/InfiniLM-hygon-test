#include "glm_resident_exact_budget_r8854.hpp"
#include "glm_model_window_r5797.hpp"
#include "glm_rope_bound_r4279.hpp"
#include "glm_transient_staging_r5680.hpp"
#include "glm_rows32_policy_r5731.hpp"
#include "glm_exact_banks_r4188.hpp"
#include "glm_shared_rope_r3783.hpp"
#include "glm_decode_leases_r1354.hpp"
#include "lease32_policy_r9205.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "glm_moe_dsa_projected_text_model_r22.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <unordered_map>
namespace rt=infinilm::runtime;namespace model=infinilm::models::glm_moe_dsa;
namespace infinilm::models::glm_moe_dsa {void glm_indexer_policy_r998(rt::Runtime&,bool);void glm_indexer_policy_clear_r998(rt::Runtime&);}
namespace{
thread_local std::string error,serialized;
struct Context{glm_model_resident_r5640::Mask resident_mask_r5640{};uint64_t resident_generation_r5640=0;bool bounded_rope_r4279=false,shared_control_r4279=false;uint64_t control_calls_r4279=0,position_checks_r4279=0;std::shared_ptr<rt::Runtime>owner;std::unique_ptr<model::TextModelProjectedR22>module;std::unique_ptr<model::transient_r5680::Scratch>transient_scratch_r5688;size_t rows_r5688=0;infinicclComm_t world=nullptr,tp=nullptr,control=nullptr;std::set<std::string>loaded;int stage=0,pp=4;bool ready=false;bool decode_leases_r1356=false;bool selection_r1406=false,bypass_r1406=false,owned_prefix_r1669=false;size_t capacity_r1406=0;};
struct Lease32StateR9205{bool on=false;std::array<uint64_t,13>counts{};};
thread_local std::map<Context*,Lease32StateR9205>lease32_states_r9205;
bool lease32_on_r9205(Context*c){auto i=lease32_states_r9205.find(c);return i!=lease32_states_r9205.end()&&i->second.on;}
void lease32_record_r9205(Context*c,size_t tokens,size_t count,const std::vector<model::NativeAttentionProjectedR22::Request>&metadata,int all_positions,bool enabled,const model::decode_leases_r1354::Stats&stats){auto i=lease32_states_r9205.find(c);if(i==lease32_states_r9205.end())return;auto&v=i->second.counts;++v[0];bool large=tokens==count&&count>16&&count<=32&&!all_positions;for(auto q:metadata)large=large&&q.tokens==1&&q.past>0;v[1]+=large;v[2]+=large&&enabled;v[3]+=enabled&&count<=16;if(enabled){v[4]+=stats.success_fences_deferred;v[5]+=stats.retired_groups;v[6]+=stats.retired_tensors;v[7]+=stats.logical_bytes;v[8]=std::max(v[8],uint64_t(stats.peak_logical_bytes));v[9]+=stats.budget_drains;v[10]+=stats.final_drains;v[11]+=stats.exception_drains;v[12]+=large;}}
std::map<int,std::shared_ptr<rt::Runtime>>&owners(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>pool;return pool;}
void drain(Context*c){if(c&&c->owner)try{c->owner->syncStream();}catch(...){std::terminate();}}
void producer(void*s){if(cudaStreamSynchronize(static_cast<cudaStream_t>(s))!=cudaSuccess)throw std::runtime_error("GLM full-model producer failed");}
}
extern "C" const char*modern_glm_model_error(){return error.c_str();}
extern "C" size_t modern_glm_model_id_size(){return sizeof(infinicclUniqueId);}
extern "C" int modern_glm_model_id(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*modern_glm_model_create(int device,int stage,size_t rows,size_t capacity,const char*cfg,const void*world_id,const void*tp_id,const void*control_id){std::unique_ptr<Context>c;try{
 const char*setting=std::getenv("GLM_PP_SIZE_R3603");int pp=4;if(setting){if(std::string(setting)=="2")pp=2;else if(std::string(setting)=="3")pp=3;else if(std::string(setting)!="4")throw std::invalid_argument("GLM PP size must be exactly2,3 or4");}if(device<0||device>=8||stage<0||stage>=pp)throw std::invalid_argument("GLM full-model requires TP8 PP2/3/4");c=std::make_unique<Context>();auto flag4279=[](const char*name){const char*p=std::getenv(name);if(!p||std::string(p)=="0")return false;if(std::string(p)=="1")return true;throw std::invalid_argument("4279 flags require0or1");};c->bounded_rope_r4279=flag4279("GLM_BOUNDED_ROPE_R4279");c->shared_control_r4279=flag4279("GLM_SHARED_CONTROL_R4279");auto&p=owners()[device];if(!p){p=rt::Runtime::create({rt::Device::Type::kHygon,device});p->enableOwnerDrainFailStop();}c->owner=p;c->stage=stage;c->pp=pp;c->capacity_r1406=capacity;c->rows_r5688=rows;auto&r=*p;infini::rt::DeviceGuard guard{r.device().type(),device,true};
 infinicclUniqueId world{},tp{},control{};std::memcpy(&world,world_id,sizeof(world));std::memcpy(&tp,tp_id,sizeof(tp));std::memcpy(&control,control_id,sizeof(control));
 if(infinicclCommInitRank(&c->world,8*pp,world,stage*8+device)!=infinicclSuccess)throw std::runtime_error("GLM world init failed");
 if(infinicclCommInitRank(&c->tp,8,tp,device)!=infinicclSuccess)throw std::runtime_error("GLM TP init failed");
 if(c->shared_control_r4279)c->control=c->world;else if(infinicclCommInitRank(&c->control,8*pp,control,stage*8+device)!=infinicclSuccess)throw std::runtime_error("GLM control init failed");
 infinilm::engine::distributed::RankInfo rank(r.device());rank.tp_size=8;rank.tp_rank=device;rank.comm=c->tp;rank.pp_size=pp;rank.pp_stage=stage;rank.world_size=8*pp;rank.world_rank=stage*8+device;rank.world_comm=c->world;
 model::rope_bound_r4279::Scope bounded4279(r,c->bounded_rope_r4279,capacity);
 c->module=std::make_unique<model::TextModelProjectedR22>(r,nlohmann::json::parse(cfg),rank,rows,capacity,&c->bypass_r1406,&c->owned_prefix_r1669);r.syncStream();return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" const char*modern_glm_model_keys(void*raw){serialized=nlohmann::json(static_cast<Context*>(raw)->module->state_dict_keys()).dump();return serialized.c_str();}
extern "C" int modern_glm_model_load_batch(void*raw,const char*descriptors){auto*c=static_cast<Context*>(raw);try{
 if(c->ready)throw std::logic_error("GLM model weights already finalized");auto values=nlohmann::json::parse(descriptors);std::unordered_map<std::string,rt::Tensor>sources;
 for(auto&item:values){std::string name=item.at("name");int type=item.at("dtype");auto shape=item.at("shape").get<rt::Shape>();auto p=reinterpret_cast<void*>(item.at("pointer").get<uintptr_t>());if(type<0||type>2||!p||sources.count(name))throw std::invalid_argument("GLM batch source differs");sources.emplace(name,rt::Tensor::from_blob(p,shape,type==0?rt::DataType::kFloat32:type==1?rt::DataType::kBFloat16:rt::DataType::kUInt8,{rt::Device::Type::kCpu,0}));}
 c->module->load_parameters_no_sync(*c->owner,sources,true);c->owner->syncStream();for(auto&item:sources)c->loaded.insert(item.first);return 0;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_finish(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->ready)throw std::logic_error("GLM model already finalized");for(auto&name:c->module->state_dict_keys())if(!c->loaded.count(name))throw std::runtime_error("Missing GLM model checkpoint parameter "+name);c->module->process_weights_after_loading(*c->owner,false);c->owner->syncStream();c->ready=true;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_forward(void*raw,size_t tokens,size_t count,const size_t*requests,const void*input,const void*positions,void*logits,int all_positions,void*stream){auto*c=static_cast<Context*>(raw);try{
 if(!c->ready||!count||count>128||!requests)throw std::invalid_argument("GLM full-model CAPI metadata differs");producer(stream);auto&r=*c->owner;std::vector<model::NativeAttentionProjectedR22::Request>metadata;for(size_t i=0;i<count;++i)metadata.push_back({requests[3*i],requests[3*i+1],requests[3*i+2]});
 if(c->bounded_rope_r4279){
  if(!positions||!tokens||tokens>count*c->capacity_r1406)throw std::invalid_argument("4279 bounded position buffer geometry");
  std::vector<size_t>pasts,widths;size_t n=0;for(const auto&q:metadata){if(!q.tokens||q.past>=c->capacity_r1406||q.tokens>c->capacity_r1406-q.past||n>tokens||q.tokens>tokens-n)throw std::invalid_argument("4279 bounded request metadata");n+=q.tokens;pasts.push_back(q.past);widths.push_back(q.tokens);}if(n!=tokens)throw std::invalid_argument("4279 request packed count differs");
  r.syncStream();std::vector<int64_t>host_positions(tokens);r.memcpyD2H(host_positions.data(),positions,tokens*sizeof(int64_t));r.syncStream();model::rope_bound_r4279::validate_cpu_positions(c->capacity_r1406,host_positions,pasts,widths);++c->position_checks_r4279;
 }
 bool lease_decode=glm_lease32_r9205::use(c->decode_leases_r1356,lease32_on_r9205(c),tokens,count,all_positions!=0);for(const auto&q:metadata)lease_decode=lease_decode&&q.tokens==1&&q.past>0;model::decode_leases_r1354::Guard lease_frame(r,lease_decode);
 auto ids=rt::Tensor::from_blob(const_cast<void*>(input),{tokens},rt::DataType::kInt64,r.device()),pos=rt::Tensor::from_blob(const_cast<void*>(positions),{tokens},rt::DataType::kInt64,r.device());auto result=c->module->forward(r,ids,pos,metadata,all_positions!=0);
 if(c->stage+1==c->pp){if(!result.logits)throw std::runtime_error("Missing final-stage GLM logits");rt::Tensor::from_blob(logits,result.logits->shape(),result.logits->dtype(),r.device())->copy_from(r,result.logits);}const bool lease_enabled_r9205=lease_frame.enabled;model::decode_leases_r1354::Stats lease_stats_r9205{};if(lease_frame.enabled)lease_stats_r9205=lease_frame.finish();else r.syncStream();lease32_record_r9205(c,tokens,count,metadata,all_positions,lease_enabled_r9205,lease_stats_r9205);return 0;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_control(void*raw,size_t count,void*values,void*stream){auto*c=static_cast<Context*>(raw);try{producer(stream);auto&r=*c->owner;if(c->shared_control_r4279)r.syncStream();auto x=rt::Tensor::from_blob(values,{count},rt::DataType::kFloat32,r.device());infinilm::engine::distributed::allreduce_(r,x,x,infinicclSum,c->control);r.syncStream();++c->control_calls_r4279;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_reset(void*raw){auto*c=static_cast<Context*>(raw);bool verified=false;try{
 if(!c||!c->module)throw std::invalid_argument("5688 reset context");auto&r=*c->owner;r.assertOwnerThread();int d=-1;
 if(cudaGetDevice(&d)!=cudaSuccess||d!=r.device().index()||r.isGraphRecording())throw std::invalid_argument("5688 reset owner/device/eager");
 cudaStreamCaptureStatus capture;if(cudaStreamIsCapturing(static_cast<cudaStream_t>(r.stream()),&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("5688 reset capture");
 verified=true;c->module->reset(r);if(c->transient_scratch_r5688)c->transient_scratch_r5688->reset();return 0;
 }catch(const std::exception&e){error=e.what();if(verified)drain(c);return 1;}}
extern "C" int modern_glm_model_indexer_policy_r998(void*raw,int enabled){auto*c=static_cast<Context*>(raw);try{if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("998 policy context/value");if(c->bypass_r1406&&!enabled)throw std::logic_error("1406 disable feature bypass after reset before disabling selection policy");model::glm_indexer_policy_r998(*c->owner,enabled!=0);c->selection_r1406=enabled!=0;return 0;}catch(const std::exception&e){error=e.what();if(c)drain(c);return 1;}}
extern "C" int modern_glm_model_destroy(void*raw){auto*c=static_cast<Context*>(raw);drain(c);if(modern_glm_fp8_runtime_clear_r809(c->owner.get())){error=modern_glm_fp8_runtime_error_r809();return 1;}model::glm_indexer_policy_clear_r998(*c->owner);glm_window_r5795::close_after_drain(*c->owner);c->module.reset();drain(c);c->transient_scratch_r5688.reset();drain(c);if((!c->shared_control_r4279&&infinicclCommDestroy(c->control)!=infinicclSuccess)||infinicclCommDestroy(c->tp)!=infinicclSuccess||infinicclCommDestroy(c->world)!=infinicclSuccess)return 1;lease32_states_r9205.erase(c);delete c;return 0;}
extern "C" int modern_glm_model_shutdown(){try{for(auto&item:owners()){infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,item.first,true};item.second->syncStream();item.second.reset();}owners().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}

// Experimental control is local to one TP stage. Numerical model code is unchanged.
extern "C" int modern_glm_model_tp_control(void*raw,size_t count,void*values,void*stream){auto*c=static_cast<Context*>(raw);try{producer(stream);auto&r=*c->owner;auto x=rt::Tensor::from_blob(values,{count},rt::DataType::kFloat32,r.device());infinilm::engine::distributed::allreduce_(r,x,x,infinicclSum,c->tp);r.syncStream();return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}

extern "C" int modern_glm_model_decode_leases_policy_r1356(void*raw,int enabled){auto*c=static_cast<Context*>(raw);try{if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("1356 policy context/value");c->owner->assertOwnerThread();if(model::decode_leases_r1354::active(*c->owner))throw std::logic_error("1356 policy during frame");c->owner->syncStream();c->decode_leases_r1356=enabled!=0;return 0;}catch(const std::exception&e){error=e.what();if(c)drain(c);return 1;}}

extern "C" int modern_glm_model_feature_bypass_r1406(void*raw,int enabled){auto*c=static_cast<Context*>(raw);try{
 if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("1406 feature policy context/value");c->owner->assertOwnerThread();
 if(model::decode_leases_r1354::active(*c->owner)||!c->module->all_sequences_empty(*c->owner))throw std::logic_error("1406 feature policy requires reset and idle owner");
 if(enabled&&(c->capacity_r1406>2048||!c->selection_r1406))throw std::invalid_argument("1406 requires configuredcapacity<=topk2048 and selection policy enabled");
 if(!enabled&&c->owned_prefix_r1669)throw std::logic_error("1669 disable owned prefix policy before disabling feature bypass");
 c->owner->syncStream();c->bypass_r1406=enabled!=0;return 0;
 }catch(const std::exception&e){error=e.what();if(c)drain(c);return 1;}}

extern "C" int modern_glm_model_owned_prefix_policy_r1669(void*raw,int enabled){auto*c=static_cast<Context*>(raw);try{
 if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("1669 owned prefix context/value");c->owner->assertOwnerThread();
 if(model::decode_leases_r1354::active(*c->owner)||!c->module->all_sequences_empty(*c->owner))throw std::logic_error("1669 owned prefix policy requires reset and idle owner");
 if(enabled&&(!c->selection_r1406||!c->bypass_r1406||c->capacity_r1406>2048))throw std::invalid_argument("1669 owned prefix requires bounded all-visible feature bypass");
 c->owner->syncStream();c->owned_prefix_r1669=enabled!=0;return 0;
 }catch(const std::exception&e){error=e.what();if(c)drain(c);return 1;}}

namespace infinilm::models::glm_moe_dsa {
void glm_fp8_batch_policy_r3068(rt::Runtime&,bool);
void glm_fp8_batch_stats_r3068(rt::Runtime&,uint64_t*,size_t);
}
extern "C" int modern_glm_model_batch_cache_policy_r3068(void*raw,int enabled) {
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try {
  if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("3068 policy context/value");
  auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))
   throw std::logic_error("3068 policy requires reset and idle owner outside graph recording");
  r.syncStream();model::glm_fp8_batch_policy_r3068(r,enabled!=0);return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}
extern "C" int modern_glm_model_batch_cache_stats_r3068(void*raw,uint64_t*out,size_t count) {
 auto*c=static_cast<Context*>(raw);
 try {
  if(!c||!c->ready||!out||count!=5)throw std::invalid_argument("3068 stats context/output/count");
  c->owner->assertOwnerThread();model::glm_fp8_batch_stats_r3068(*c->owner,out,count);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}

// Explicit idle-owner operation; default serving behavior is unchanged.
extern "C" int modern_glm_idle_trim_r3771(void*raw,uint64_t*out,size_t count){
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try{
  if(!c||!c->ready||!out||count!=6)throw std::invalid_argument("3771 trim context/output/count");
  auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||r.graphPinnedMemoryBytes()!=0||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("3771 trim requires reset eager idle owner without graph pins");
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};r.syncStream();size_t before=0,after=0,total=0,total_after=0;
  if(cudaMemGetInfo(&before,&total)!=cudaSuccess)throw std::runtime_error("3771 before cudaMemGetInfo");
  r.trimMemory();r.syncStream();
  if(cudaMemGetInfo(&after,&total_after)!=cudaSuccess||total!=total_after)throw std::runtime_error("3771 after cudaMemGetInfo");
  const uint64_t values[]={before,after,total,uint64_t(r.device().index()),uint64_t(c->stage),uint64_t(c->pp)};std::memcpy(out,values,sizeof(values));return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}

extern "C" int modern_glm_shared_rope_stats_r3783(void*raw,uint64_t*out,size_t count){auto*c=static_cast<Context*>(raw);try{if(!c||!out||count!=5)throw std::invalid_argument("3783 stats arguments");auto v=model::shared_rope_r3783::stats(*c->owner);std::memcpy(out,v.data(),v.size()*sizeof(uint64_t));return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" uint64_t modern_glm_shared_rope_live_r3783(){uint64_t n=0;for(const auto&v:model::shared_rope_r3783::pool)n+=!v.second.expired();return n;}

extern "C" const char*modern_glm_partition_report_r3854(void*raw){auto*c=static_cast<Context*>(raw);try{if(!c||!c->module)throw std::invalid_argument("3854 context");c->owner->assertOwnerThread();serialized=nlohmann::json{{"PP",c->pp},{"stage",c->stage},{"device",c->owner->device().index()},{"begin",c->module->layer_begin()},{"end",c->module->layer_end()},{"first_stage",c->module->first_stage()},{"last_stage",c->module->last_stage()},{"weights_ready",c->ready},{"keys",c->module->state_dict_keys()}}.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}

// Separate constructor-only operation. The ready-only serving trim is unchanged.
extern "C" int modern_glm_unloaded_trim_r4052(void*raw,uint64_t*out,size_t count){
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try{
  if(!c||!out||count!=6||c->ready||!c->loaded.empty())throw std::invalid_argument("4052 trim requires constructed context before any weight loading");
  auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||r.graphPinnedMemoryBytes()!=0||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("4052 constructor trim requires idle eager owner");
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};r.syncStream();size_t before=0,after=0,total=0,total_after=0;
  if(cudaMemGetInfo(&before,&total)!=cudaSuccess)throw std::runtime_error("4052 before cudaMemGetInfo");r.trimMemory();r.syncStream();
  if(cudaMemGetInfo(&after,&total_after)!=cudaSuccess||total!=total_after)throw std::runtime_error("4052 after cudaMemGetInfo");
  const uint64_t values[]={before,after,total,uint64_t(r.device().index()),uint64_t(c->stage),uint64_t(c->pp)};std::memcpy(out,values,sizeof(values));return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}

extern "C" const char*modern_glm_constructor_census_r4158(void*raw){
 auto*c=static_cast<Context*>(raw);try{if(!c||!c->module||c->ready||!c->loaded.empty())throw std::invalid_argument("4158 requires unloaded context");c->owner->assertOwnerThread();serialized=c->module->constructor_memory_census_r4158(*c->owner).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}
}

extern "C" int modern_glm_exact_banks_stats_r4188(void*raw,uint64_t*out,size_t count){auto*c=static_cast<Context*>(raw);try{if(!c||!out||count!=4)throw std::invalid_argument("4188 stats context/count");c->owner->assertOwnerThread();const auto v=model::exact_banks_r4188::stats(*c->owner);std::memcpy(out,v.data(),sizeof(uint64_t)*4);return 0;}catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int modern_glm_service_memory_stats_r4279(void*raw,uint64_t*out,size_t n){auto*c=static_cast<Context*>(raw);try{
 if(!c||!out||n!=9)throw std::invalid_argument("4279 stats context/count");auto&r=*c->owner;r.assertOwnerThread();const auto a=model::rope_bound_r4279::stats(r);uint64_t values[9]={uint64_t(c->bounded_rope_r4279),uint64_t(c->shared_control_r4279),uint64_t(c->capacity_r1406),c->control_calls_r4279,c->position_checks_r4279,a[0],a[1],a[2],a[3]};std::memcpy(out,values,sizeof(values));return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int modern_glm_projected_host_counts_r4504(int device,uint64_t*out,size_t n){return infinilm::models::glm_moe_dsa::projected_host_r4504::stats(device,out,n);}

// Proposed private Model API. DefaultOFF; no constructor change, extraRuntime
// or callerstream. Caller must agree exactglobalmask across this TP8 stage.
extern "C" int modern_glm_model_resident_mask_r5640(void*raw,const uint8_t*bits,size_t count,int memory_trace){auto*c=static_cast<Context*>(raw);bool owner_verified=false;try{
 if(!c||!c->ready||c->pp!=2||(memory_trace!=0&&memory_trace!=1)||c->capacity_r1406!=640)throw std::invalid_argument("5640 ready/PP2/cap640/trace");
 auto&r=*c->owner;r.assertOwnerThread();int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=r.device().index()||r.isGraphRecording())throw std::invalid_argument("5640 owner/currentdevice/eager");
 cudaStreamCaptureStatus capture;if(cudaStreamIsCapturing(static_cast<cudaStream_t>(r.stream()),&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("5640 runtime capture");
 owner_verified=true;if(!c->module->all_sequences_empty(r))throw std::invalid_argument("5640 policy onlydrainedreset EMPTY");
 auto next=glm_rows32_r5731::mask(bits,count,c->module->layer_begin(),c->module->layer_end(),c->rows_r5688);if(c->transient_scratch_r5688&&c->transient_scratch_r5688->enabled()&&glm_model_resident_r5640::selected(next))throw std::invalid_argument("5688 resident and transient mutually exclusive");r.syncStream();
 size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess)throw std::runtime_error("5640 memory query");
 if(glm_model_resident_r5640::selected(next))glm_model_resident_r5640::admit(free,glm_resident_exact_r8854::admission_bytes(c->rows_r5688,glm_model_resident_r5640::selected(next)));
 auto staged=c->module->resident_prepare_r5640(r,next);r.syncStream();
 if(cudaMemGetInfo(&free,&total)!=cudaSuccess)throw std::runtime_error("5640 afterallocation memory");
 if(glm_model_resident_r5640::selected(next))glm_model_resident_r5640::admit(free,0);
 // No allocations/enqueue/throw during publication. Policy pointer remains
 // stable until module destruction, which precedes Context destruction.
 c->resident_mask_r5640=next;c->module->resident_publish_r5640(c->resident_mask_r5640.data(),std::move(staged),memory_trace!=0);++c->resident_generation_r5640;return 0;
 }catch(const std::exception&e){error=e.what();if(owner_verified)drain(c);return 1;}}
extern "C" const char*modern_glm_model_resident_report_r5640(void*raw){auto*c=static_cast<Context*>(raw);try{if(!c||!c->ready)throw std::invalid_argument("5640 report context");c->owner->assertOwnerThread();auto q=c->module->resident_report_r5640(*c->owner);q["resident_physical_bytes_per_layer"]=glm_resident_exact_r8854::admission_bytes(c->rows_r5688,1);q["generation"]=c->resident_generation_r5640;q["Context_mask"]=c->resident_mask_r5640;q["Runtime_identity"]=reinterpret_cast<uintptr_t>(c->owner.get());serialized=q.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}

// Private Model-only policy: SAME original Runtime and hostbank; defaultOFF.
// No callerstream or Context is published until ready/EMPTY/owner/memory gates.
extern "C" int modern_glm_model_transient_policy_r5688(void*raw,int enabled){
 auto*c=static_cast<Context*>(raw);bool verified=false;try{
 if(!c||!c->ready||(enabled!=0&&enabled!=1)||c->pp!=2||!glm_rows32_r5731::transient_geometry(c->rows_r5688,enabled)||c->capacity_r1406!=640)throw std::invalid_argument("5688 policy ready/PP2/rows64/cap640/value");
 auto&r=*c->owner;r.assertOwnerThread();int d=-1;if(cudaGetDevice(&d)!=cudaSuccess||d!=r.device().index()||r.isGraphRecording())throw std::invalid_argument("5688 owner/device/eager");
 cudaStreamCaptureStatus capture;if(cudaStreamIsCapturing(static_cast<cudaStream_t>(r.stream()),&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("5688 Runtime capture");
 verified=true;if(model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::invalid_argument("5688 policy only EMPTY/drained eager owner");
 if(enabled&&(!c->selection_r1406||!c->bypass_r1406||!c->owned_prefix_r1669||glm_model_resident_r5640::selected(c->resident_mask_r5640)))throw std::invalid_argument("5688 provenhost prerequisite or resident conflict");
 r.syncStream();
 if(enabled){size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess)throw std::runtime_error("5688 admission memory query");
  constexpr size_t floor=size_t{6}<<30,physical_scratch=size_t{64}<<20;
  const bool allocated=c->transient_scratch_r5688&&c->transient_scratch_r5688->report().at("GPU_logical_bytes").get<size_t>()!=0;
  const size_t add=allocated?0:physical_scratch;if(free<floor||add>free-floor)throw std::invalid_argument("5688 sixGiB admission plus physical64MiB scratch");
  if(!c->transient_scratch_r5688){auto pending=std::make_unique<model::transient_r5680::Scratch>(r);pending->policy(true);r.syncStream();
   if(cudaMemGetInfo(&free,&total)!=cudaSuccess||free<floor)throw std::runtime_error("5688 postallocation sixGiB gate");c->transient_scratch_r5688=std::move(pending);
  }else{
   // A published Scratch already owns both banks: no additional allocation.
   // Fresh prequery above established sixGiB; policy only drains/enables.
   if(!allocated)throw std::logic_error("5688 published scratch missing banks");c->transient_scratch_r5688->policy(true);
  }
 }else if(c->transient_scratch_r5688)c->transient_scratch_r5688->policy(false);
 return 0;
 }catch(const std::exception&e){error=e.what();if(verified)drain(c);return 1;}
}
extern "C" const char*modern_glm_model_transient_report_r5688(void*raw){auto*c=static_cast<Context*>(raw);try{
 if(!c||!c->ready)throw std::invalid_argument("5688 report context");auto&r=*c->owner;r.assertOwnerThread();
 nlohmann::json q=c->transient_scratch_r5688?c->transient_scratch_r5688->report():nlohmann::json{{"enabled",false},{"busy",false},{"poison",false},{"calls",0},{"GPU_logical_bytes",0},{"pinned_slab_bytes",0}};
 q["Runtime_identity"]=reinterpret_cast<uintptr_t>(c->owner.get());q["rows"]=c->rows_r5688;q["capacity"]=c->capacity_r1406;q["stage"]=c->stage;q["PP"]=c->pp;serialized=q.dump();return serialized.c_str();
 }catch(const std::exception&e){error=e.what();return nullptr;}}

extern "C" int modern_glm_window_arm_r5795(void*raw,int round,int cohort,int enabled){try{auto*c=static_cast<Context*>(raw);if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("5795 arm context");glm_window_r5795::arm(*c->owner,round,cohort,enabled!=0);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_glm_window_read_r5795(void*raw){try{auto*c=static_cast<Context*>(raw);if(!c||!c->ready)throw std::invalid_argument("5795 read context");return glm_window_r5795::read_after_CAPI(*c->owner);}catch(const std::exception&e){error=e.what();return nullptr;}}

extern "C" int modern_glm_window_prepare_r5797(void*raw){try{auto*c=static_cast<Context*>(raw);if(!c||!c->ready)throw std::invalid_argument("5797 prepare context");c->owner->assertOwnerThread();if(model::decode_leases_r1354::active(*c->owner)||!c->module->all_sequences_empty(*c->owner))throw std::invalid_argument("5797 prepare EMPTY/drained only");glm_window_r5795::prepare(*c->owner);return 0;}catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int modern_glm_decode_lease_m32_r9205(void*raw,int on){auto*c=static_cast<Context*>(raw);try{if(!c||!c->owner||!c->module||!c->ready||(on!=0&&on!=1))throw std::invalid_argument("9205 livecontext/policy");auto&r=*c->owner;r.assertOwnerThread();if(r.isGraphRecording()||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("9205 requires EMPTY/idle/eager owner");if(on&&(c->rows_r5688!=32||c->capacity_r1406!=640||!c->decode_leases_r1356))throw std::invalid_argument("9205 only originalROWS32/CAP640/decodeleasesON");lease32_states_r9205[c].on=on;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_glm_decode_lease_m32_get_r9205(void*raw){auto*c=static_cast<Context*>(raw);try{if(!c||!c->owner||!c->module||!c->ready)throw std::invalid_argument("9205 livecontext");c->owner->assertOwnerThread();return lease32_on_r9205(c)?1:0;}catch(const std::exception&e){error=e.what();return -1;}}
extern "C" int modern_glm_decode_lease_m32_counts_r9205(void*raw,uint64_t*out,size_t n){auto*c=static_cast<Context*>(raw);try{if(!c||!c->owner||!c->module||!c->ready||!out||n!=13)throw std::invalid_argument("9205 livecontext/counters");c->owner->assertOwnerThread();auto i=lease32_states_r9205.find(c);for(size_t k=0;k<13;++k)out[k]=i==lease32_states_r9205.end()?0:i->second.counts[k];return 0;}catch(const std::exception&e){error=e.what();return 1;}}
