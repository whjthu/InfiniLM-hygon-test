#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include "config/config_factory.hpp"
#include "models/kimi_k3/kimi_k3_allocate_cache.hpp"
#include "probe-prefix-r171.inc"
#include <set>
#include <memory>
#include <thread>
#include <cstdio>
namespace infinilm::models::kimi_k3 {
namespace {
namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;using J171=nlohmann::json;
constexpr uint64_t Floor171=4ULL*1024*1024*1024;
thread_local std::string result171;
void checked171(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error("r171 CUDA operation failed");}
J171 memory171(){size_t free=0,total=0;checked171(cudaMemGetInfo(&free,&total));if(free<=Floor171)throw std::runtime_error("r171 unchanged4GiB floor");return{{"free_bytes",free},{"total_bytes",total},{"device_used_bytes",total-free},{"Native_pool_reserved_bytes","UNKNOWN_NO_PUBLIC_STATS_API"}};}
void reserve171(uint64_t extra){auto m=memory171();if(m["free_bytes"].get<uint64_t>()<=Floor171+extra)throw std::runtime_error("r171 preallocation reserve stop");}
class Binding171 final:public infinilm::weights::ModelWeights {public:void store(const std::string&n,infinilm::weights::Weight w){this->bind_weight(n,std::move(w));}};
struct Bank171{Binding171 binding;KimiK3Mxfp4MoeWeights banks;};
J171 check_bank171(Bank171&set){
 std::array<rt::Tensor,4>b{{set.banks.packed_w13,set.banks.w13_scale,set.banks.packed_w2,set.banks.w2_scale}};
 const std::array<rt::Shape,4>shapes{{{896,192,1792},{896,192,112},{896,3584,48},{896,3584,3}}};std::set<const void*>unique;size_t bytes=0;
 for(size_t k=0;k<4;++k){if(!b[k]||b[k]->shape()!=shapes[k]||b[k]->dtype()!=rt::DataType::kUInt8||!b[k]->is_contiguous()||!unique.insert(b[k]->data()).second)throw std::runtime_error("r171 actual bank geometry");bytes+=b[k]->nbytes();}
 auto leaves=set.binding.state_dict();if(leaves.size()!=5376||bytes!=491323392)throw std::runtime_error("r171 original leaf count/bytes");size_t count=0;
 struct L{const char*n;size_t bank,row,rows,width;};const std::array<L,6>spec{{{"w1.weight_packed",0,0,96,1792},{"w1.weight_scale",1,0,96,112},{"w3.weight_packed",0,96,96,1792},{"w3.weight_scale",1,96,96,112},{"w2.weight_packed",2,0,3584,48},{"w2.weight_scale",3,0,3584,3}}};
 for(size_t e=0;e<896;++e)for(auto&s:spec){auto t=leaves.at(std::to_string(e)+"."+s.n).tensor();auto expected=b[s.bank]->narrow({{0,e,1},{1,s.row,s.rows}})->squeeze(0);size_t off=(e*shapes[s.bank][1]+s.row)*s.width;
  if(t->shape()!=rt::Shape{s.rows,s.width}||t->strides()!=expected->strides()||t->data()!=expected->data()||t->data()!=b[s.bank]->data()+off||off+t->nbytes()>b[s.bank]->nbytes())throw std::runtime_error("r171 original alias/last offset");++count;
 }
 return{{"unique_storages",unique.size()},{"checked_leaf_aliases",count},{"logical_bytes",bytes},{"FP4_payload_read",false}};
}
size_t check_zero171(rt::Runtime&r,const rt::Tensor&t){
 if(!t||t->dtype()!=rt::DataType::kBFloat16||!t->is_contiguous())throw std::runtime_error("r171 cache dtype/contiguous");std::vector<unsigned char>host(8*1024*1024);size_t checked=0;
 while(checked<t->nbytes()){size_t n=std::min(host.size(),t->nbytes()-checked);r.memcpyD2H(host.data(),t->data()+checked,n);if(std::any_of(host.begin(),host.begin()+n,[](unsigned char v){return v!=0;}))throw std::runtime_error("r171 nonzero initialized cache");checked+=n;}
 return checked;
}
uint64_t generation171(rt::Runtime&r){std::lock_guard<std::mutex>g(r170_scratch::registry_mutex);auto i=r170_scratch::registry.find(r170_scratch::WeakOwner(r.shared_from_this()));if(i==r170_scratch::registry.end())throw std::runtime_error("r171 missing actual lease registry");return i->second->generation;}
struct HostMarker171{};
void lease_consumer171(rt::Runtime&r,const rt::Tensor&scratch,bool throw_host,bool test_nested){
 auto out=rt::Tensor::empty(r,scratch->shape(),scratch->dtype());if(out->data()==scratch->data())throw std::runtime_error("r171 output alias");cudaEvent_t event=nullptr;checked171(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
 try{
  bool caught=false;
  try{{r170_scratch::Lease lease(r,scratch);r.setDeviceMemoryAsync(scratch->data(),0x5a,scratch->nbytes(),r.stream());checked171(cudaMemcpyAsync(out->data(),scratch->data(),scratch->nbytes(),cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(r.stream())));checked171(cudaEventRecord(event,static_cast<cudaStream_t>(r.stream())));
    if(test_nested){bool denied=false;try{r170_scratch::Lease nested(r,scratch);}catch(const std::logic_error&){denied=true;}if(!denied)throw std::runtime_error("r171 reentrant lease accepted");}
    if(throw_host)throw HostMarker171{};
  }}catch(const HostMarker171&){caught=true;}
  if(caught!=throw_host)throw std::runtime_error("r171 wrong host unwind");
  // There was no sync between record and actual Lease destructor. Its real drain must complete the event.
  checked171(cudaEventQuery(event));std::vector<unsigned char>host(scratch->nbytes());r.memcpyD2H(host.data(),out->data(),host.size());if(std::any_of(host.begin(),host.end(),[](unsigned char v){return v!=0x5a;}))throw std::runtime_error("r171 lease consumer result");checked171(cudaEventDestroy(event));event=nullptr;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}if(event)cudaEventDestroy(event);throw;}
}
}
extern "C" const char*modern_kimi_persistent_subset_result_r171(){return result171.c_str();}
extern "C" int modern_kimi_persistent_subset_r171(const char*payload){
 J171 report={{"completed",false},{"scope","actual persistent allocation subset and actual r170 lease; not fullModel/TP32execution"},{"weight_payload_loaded",false},{"model_forward_calls",0},{"communicator_created",false},{"allocation_geometry_only",true},{"fullModel_GO",false}};
 try{
  if(!payload)throw std::invalid_argument("r171 config metadata required");int count=0;checked171(cudaGetDeviceCount(&count));if(count!=1)throw std::runtime_error("r171 one visible GPU");infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,0,true};report["before_RuntimeA"]=memory171();
  auto raw=J171::parse(payload);raw["infinilm_text_only"]=true;raw["infinilm_kimi_global_tp_adapter"]=true;raw["text_config"]["infinilm_kimi_global_tp_adapter"]=true;raw["infinilm_kimi_engine_adapter"]=true;raw["infinilm_kimi_compact_cache_r4974"]=false;raw["text_config"]["infinilm_kimi_compact_cache_r4974"]=false;raw["dtype"]="bfloat16";raw["text_config"]["dtype"]="bfloat16";raw["quantization_config"]=nullptr;raw.erase("lm_head_output_dtype");raw["text_config"].erase("lm_head_output_dtype");auto cfg=infinilm::config::ConfigFactory::createConfig(raw.dump());
  if(cfg->get<size_t>("num_hidden_layers")!=93||cfg->get<size_t>("num_attention_heads")!=96||cfg->get<size_t>("head_dim")!=192||cfg->get_config_json().at("linear_attn_config").at("num_heads")!=96)throw std::runtime_error("r171 exact cache geometry");
  auto ownerA=rt::Runtime::create({rt::Device::Type::kHygon,0});auto&a=*ownerA;a.enableOwnerDrainFailStop();report["after_RuntimeA"]=memory171();
  std::vector<std::unique_ptr<Bank171>>banks;report["bank_allocation_endpoints"]=J171::array();size_t aliases=0;
  for(size_t layer=0;layer<92;++layer){reserve171(3ULL*1024*1024*1024);auto b=std::make_unique<Bank171>();b->banks=make_packed_subset_r171(a,896,3584,96,0,32,[&](const std::string&n,infinilm::weights::Weight w){b->binding.store(n,std::move(w));});auto info=check_bank171(*b);aliases+=info["checked_leaf_aliases"].get<size_t>();banks.push_back(std::move(b));report["bank_allocation_endpoints"].push_back({{"actual_banks",layer+1},{"memory",memory171()}});if(layer==0||layer==31||layer==63||layer==91)std::printf("r171 allocated actual bank sets %zu\n",layer+1);}
  report["packed_sets"]=92;report["packed_unique_storages"]=368;report["checked_leaf_aliases"]=aliases;report["packed_logical_bytes"]=45201752064ULL;report["after_all92_banks"]=memory171();
  reserve171(2ULL*1024*1024*1024);en::RankInfo rank{};rank.device={rt::Device::Type::kHygon,0};rank.tp_size=32;rank.tp_rank=0;rank.pp_size=1;rank.pp_stage=0;rank.comm=nullptr;infinilm::cache::PagedKVCacheConfig cc(68,256,16);
  auto caches=kimi_k3_allocate_cache_tensors(a,rank,&cc,cfg,infinilm::backends::AttentionBackend::PAGED_ATTN);a.syncStream();size_t mla=0,kda=0,cachebytes=0,zero_bytes=0;
  if(caches.kv_cache_tensors.size()!=93||caches.conv_state_tensors.size()!=93||caches.ssm_state_tensors.size()!=93)throw std::runtime_error("r171 cache vectors");
  for(size_t i=0;i<93;++i){auto kv=caches.kv_cache_tensors[i],conv=caches.conv_state_tensors[i],ssm=caches.ssm_state_tensors[i];if(kv){++mla;if(kv->shape()!=rt::Shape{2,68,256,3,192}||conv||ssm)throw std::runtime_error("r171 MLA cache shape");cachebytes+=kv->nbytes();zero_bytes+=check_zero171(a,kv);}else{++kda;if(!conv||!ssm||conv->shape()!=rt::Shape{17,1152,3}||ssm->shape()!=rt::Shape{17,3,128,128})throw std::runtime_error("r171 KDA cache shape");cachebytes+=conv->nbytes()+ssm->nbytes();zero_bytes+=check_zero171(a,conv)+check_zero171(a,ssm);}}
  if(mla!=24||kda!=69||cachebytes!=1086011136||zero_bytes!=cachebytes)throw std::runtime_error("r171 full cache count/zeroing");report["cache"]={{"MLA_layers",mla},{"KDA_layers",kda},{"logical_bytes",cachebytes},{"all_bytes_zero_readback",zero_bytes},{"actual_factory_comm_null",true}};report["after_all93_cache"]=memory171();
  std::vector<rt::Tensor>fields;for(size_t i=0;i<92;++i){auto t=r170_scratch::shared_workspace(a);if(i&&t->shared_from_this()!=fields[0]->shared_from_this())throw std::runtime_error("r171 layer workspace not shared");fields.push_back(t);}auto genA=generation171(a);report["workspace"]={{"layer_fields",fields.size()},{"unique_storage",1},{"logical_bytes",fields[0]->nbytes()},{"ownerA_generation",genA},{"not_actual_MoE_objects",true}};report["after_shared92_fields"]=memory171();
  reserve171(32*1024*1024);auto ownerB=rt::Runtime::create({rt::Device::Type::kHygon,0});auto&b=*ownerB;b.enableOwnerDrainFailStop();auto scratchB=r170_scratch::shared_workspace(b);if(scratchB->data()==fields[0]->data()||generation171(b)==genA)throw std::runtime_error("r171 crossowner storage/generation alias");bool cross_denied=false;try{r170_scratch::Lease wrong(a,scratchB);}catch(const std::logic_error&){cross_denied=true;}if(!cross_denied)throw std::runtime_error("r171 crossowner lease accepted");
  bool offthread_denied=false;std::thread wrong_thread([&]{try{r170_scratch::Lease wrong(a,fields[0]);}catch(const std::logic_error&){offthread_denied=true;}});wrong_thread.join();if(!offthread_denied)throw std::runtime_error("r171 foreign owner thread accepted");
  lease_consumer171(a,fields[0],false,true);lease_consumer171(a,fields[0],true,false);lease_consumer171(b,scratchB,false,false);report["lease_tests"]={{"normal_drain_event_and_full_pattern",true},{"host_exception_drain_event_and_full_pattern",true},{"reentrant_rejected",true},{"crossowner_rejected",cross_denied},{"wrong_owner_thread_rejected_before_GPU",offthread_denied},{"ownerB_isolated",true},{"device_fault_injected",false},{"crossstream_guard","source-only: immutable Runtime stream, no private mutation"}};report["after_lease_tests"]=memory171();
  std::weak_ptr<rt::Runtime>weakA=ownerA,weakB=ownerB;std::weak_ptr<rt::TensorImpl>weakTA=fields[0]->shared_from_this(),weakTB=scratchB->shared_from_this();scratchB.reset();ownerB.reset();if(!weakB.expired()||!weakTB.expired())throw std::runtime_error("r171 ownerB retained by registry");report["after_RuntimeB_destroy"]=memory171();
  fields.clear();caches.kv_cache_tensors.clear();caches.conv_state_tensors.clear();caches.ssm_state_tensors.clear();banks.clear();a.syncStream();report["after_subset_objects_destroy"]=memory171();ownerA.reset();if(!weakA.expired()||!weakTA.expired())throw std::runtime_error("r171 ownerA retained by registry");report["after_RuntimeA_destroy"]=memory171();
  auto ownerC=rt::Runtime::create({rt::Device::Type::kHygon,0});ownerC->enableOwnerDrainFailStop();auto scratchC=r170_scratch::shared_workspace(*ownerC);auto genC=generation171(*ownerC);if(genC==genA)throw std::runtime_error("r171 generation reused");std::weak_ptr<rt::Runtime>weakC=ownerC;std::weak_ptr<rt::TensorImpl>weakTC=scratchC->shared_from_this();scratchC.reset();ownerC.reset();if(!weakC.expired()||!weakTC.expired())throw std::runtime_error("r171 ownerC retained");report["after_RuntimeC_destroy"]=memory171();report["weak_owner_tensor_release"]={{"ownerA_expired",true},{"tensorA_expired",true},{"ownerB_expired",true},{"tensorB_expired",true},{"ownerC_expired",true},{"tensorC_expired",true},{"new_owner_generation_different",true},{"raw_address_reuse_forced",false}};
  report["logical_subset_bytes"]=46291641856ULL;report["actual_pool_reserved"]="UNKNOWN";report["GPU_measured_fullModel_fit"]="UNKNOWN_STOP";report["operator_load_CCL_highwater"]="UNKNOWN_NOT_TESTED";report["performance"]="UNKNOWN; r170 per-I96 explicit sync remains";report["normal_Runtime_close"]=true;report["completed"]=true;result171=report.dump();return 0;
 }catch(const std::exception&e){report["error"]=e.what();result171=report.dump();return 1;}
}
} // namespace infinilm::models::kimi_k3
