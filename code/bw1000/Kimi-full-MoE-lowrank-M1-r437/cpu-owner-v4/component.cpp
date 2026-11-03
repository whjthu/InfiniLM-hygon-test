#include "owner_affinity.hpp"
#include "private_moe.hpp"
#include "projection.hpp"
#include "config/config_factory.hpp"
#include "engine/distributed/communication_group.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sched.h>
#include <thread>
#include <unordered_map>

extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
extern "C" void modern_kimi_set_grouped_prefill_private_r437(int);
extern "C" void modern_kimi_set_tiled_w13_r1861_private_r437(int);
namespace {
using J=nlohmann::json;namespace rt=infinilm::runtime;
namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
using Clock=std::chrono::steady_clock;
thread_local std::string serialized;
double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
struct Barrier{std::mutex m;std::condition_variable cv;size_t count=0,generation=0;void wait(){std::unique_lock<std::mutex>lock(m);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
std::vector<int> affinity(){cpu_set_t mask;CPU_ZERO(&mask);if(sched_getaffinity(0,sizeof(mask),&mask))throw std::runtime_error("437 owner affinity read");std::vector<int>x;for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&mask))x.push_back(i);return x;}
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
double fp(uint16_t bits){uint32_t u=uint32_t(bits)<<16;float v;std::memcpy(&v,&u,4);return v;}
void write(const std::string&p,const J&q){std::ofstream f(p);f<<q.dump(2)<<'\n';if(!f)throw std::runtime_error("437 per-owner report persistence");}
}
extern "C" const char*modern_kimi_lowrank_full_moe_result_r437(){return serialized.c_str();}
extern "C" int modern_kimi_lowrank_full_moe_r437(const char*payload){
 J result={{"completed",false},{"GPU_Model_qualification",false},{"scope","Layer1 actual TP16 PP1 C1 complete MoE, historical teacher first row; not current PP2 exact-scale allocation or Model speed"}};
 try{
  if(!payload)throw std::invalid_argument("437 null payload");const J q=J::parse(payload);
  for(const auto&policy:std::array<std::pair<const char*,const char*>,6>{{{"INFINILM_KIMI_SPEC_MATH_R3676","0"},{"INFINILM_KIMI_MOE_WORKSPACE_R4665","1"},{"INFINILM_KIMI_MOE_EXECUTION_R4665","0"},{"INFINILM_KIMI_I192_GROUPED_R8669Y","0"},{"INFINILM_KIMI_I192_COMPACT_R8669ZC","1"},{"INFINILM_OWNER_CPU_WIDTH_R2392","4"}}}){const char*v=std::getenv(policy.first);if(!v||std::string(v)!=policy.second)throw std::invalid_argument(std::string("437 frozen policy ")+policy.first);}
  const int node=q.at("node_rank");if(node<0||node>1||q.at("layer").get<int>()!=1)throw std::invalid_argument("437 TP16 node0/1 and actual layer1 only");
  auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();
  if(normalized.value("infinilm_kimi_long_agent_r179",false))throw std::invalid_argument("437 PP1 component cannot claim r201 PP2 exact-scale long factory");
  const auto entry=affinity();if(entry.size()<32)throw std::invalid_argument("437 eight real owner Runtime constructors require entrymask32+");
  std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;
  en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=2;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr");topology.master_port=q.at("master_port");
  {
   en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
   for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{
    J row={{"node_rank",node},{"local_rank",rank},{"arms",J::array()},{"completed",false}};
    const std::string stem=q.at("raw_directory").get<std::string>()+"/rank"+std::to_string(rank);
    auto save=[&]{write(stem+".json",row);};
    try{
     row["entry_mask"]=kimi_r437_owner::mask();bool restored=false;kimi_r437_owner::Pin pin(entry,rank,&restored);row["pinned_mask"]=kimi_r437_owner::mask();
     auto info=group.get_rank_info(rank);if(info.tp_size!=16||info.tp_rank!=node*8+rank||!info.comm||info.pp_size!=1||info.pp_stage!=0)throw std::runtime_error("437 actual TP16/PP1 RankInfo required");
     infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();
     row["global_tp_rank"]=info.tp_rank;row["TP"]=info.tp_size;row["PP"]=info.pp_size;row["owner_mask"]=affinity();if(row["owner_mask"].get<std::vector<int>>()!=kimi_r437_owner::selected(entry,rank))throw std::runtime_error("437 Runtime owner width4 not inherited entrymask");
     {
      auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);
      auto original=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});
      auto candidate=std::make_unique<km::KimiK3MoER437>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});
      std::unordered_map<std::string,rt::Tensor>weights;
      for(const auto&v:q.at("weights")){
       const int t=v.at("dtype");if(t<0||t>2)throw std::invalid_argument("437 original checkpoint dtype only");auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;
       std::string key=v.at("name");auto tensor=rt::Tensor::from_blob(reinterpret_cast<void*>(v.at("pointer").get<uintptr_t>()),v.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0});
       if(!weights.emplace(key,tensor).second)throw std::invalid_argument("437 duplicate checkpoint descriptor");
      }
      const auto keys=original->state_required_keys();if(keys!=candidate->state_required_keys()||keys.size()!=weights.size())throw std::runtime_error("437 exact required checkpoint key set");for(const auto&key:keys)if(!weights.count(key))throw std::runtime_error("437 missing "+key);
      auto load=Clock::now();original->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*original);candidate->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*candidate);row["two_private_instances_weight_load_ms"]=ms(load);row["loaded_checkpoint_keys"]=keys.size();weights.clear();
      // Decode flags3 in the pinned public AR forward means grouped=true/tiled=false.
      modern_kimi_set_grouped_prefill(1);modern_kimi_set_tiled_w13_r1861(0);modern_kimi_set_grouped_prefill_private_r437(1);modern_kimi_set_tiled_w13_r1861_private_r437(0);
      auto storage=rt::Tensor::empty(r,{1,2,7168},rt::DataType::kBFloat16);std::vector<uint16_t>input_original(2*7168,0x3f80);
      std::memcpy(input_original.data(),reinterpret_cast<void*>(q.at("input_pointer").get<uintptr_t>()),7168*2);r.memcpyH2D(storage->data(),input_original.data(),storage->nbytes(),false);r.syncStream();auto input=storage->narrow({{1,0,1}});
      std::vector<uint16_t>reference;const std::array<int,6>plan{0,1,0,1,1,0};
      for(size_t arm=0;arm<plan.size();++arm){
       phase.wait();auto scalar=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float one=1,total=0;r.memcpyH2D(scalar->data(),&one,4,false);r.syncStream();en::allreduce_(r,scalar,scalar,infinicclSum,info.comm);r.syncStream();r.memcpyD2H(&total,scalar->data(),4);if(total!=16)throw std::runtime_error("437 actual 16 participant barrier");
       kimi_r437::selected=plan[arm]!=0;const auto before=kimi_r437::hits;const auto begin=Clock::now();auto y=plan[arm]?candidate->forward(r,input):original->forward(r,input);r.syncStream();const double native_ms=ms(begin);
       std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());
       J item={{"arm",arm},{"mode",plan[arm]},{"warm",arm<2},{"full_MoE_forward_and_completion_ms",native_ms},{"outer_completion_included",true},{"global_barrier_sum",total},{"down_up_selected_delta",{kimi_r437::hits[0]-before[0],kimi_r437::hits[1]-before[1]}},{"raw_output_elements",bits.size()}};
       std::ofstream output(stem+"-arm"+std::to_string(arm)+".bf16",std::ios::binary);output.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);if(!output)throw std::runtime_error("437 output persistence");output.close();
       if(arm==0)reference=bits;double d2=0,b2=0,maxabs=0;size_t nf=0,different=0;for(size_t i=0;i<bits.size();++i){double a=fp(reference.at(i)),b=fp(bits[i]),d=a-b;nf+=!std::isfinite(a)||!std::isfinite(b);d2+=d*d;b2+=a*a;maxabs=std::max(maxabs,std::abs(d));different+=reference[i]!=bits[i];}
       std::vector<uint16_t>checked(input_original.size());r.memcpyD2H(checked.data(),storage->data(),storage->nbytes());
       item["nonfinite"]=nf;item["relative_RMS_vs_first_OFF"]=std::sqrt(d2/std::max(b2,1e-30));item["max_abs_vs_first_OFF"]=maxabs;item["bit_differences_diagnostic_only"]=different;item["input_and_tail_canary_unchanged"]=checked==input_original;
       y.reset();r.syncStream();item["checked_call_including_output_readback_save_release_ms"]=ms(begin);row["arms"].push_back(item);save();
       // Full raw + metrics persist first; .10 is unexplained-drift STOP, not a quality bound.
       if(nf||checked!=input_original||bits.size()!=7168||item["down_up_selected_delta"]!=J::array({plan[arm],plan[arm]}))throw std::runtime_error("437 finite/canary/output/actual two-projection hit failure");
       if(item["relative_RMS_vs_first_OFF"].get<double>()>.10)throw std::runtime_error("437 unexplained full-MoE numerical drift; raw retained");
      }
      kimi_r437::selected=false;candidate.reset();original.reset();r.syncStream();row["both_modules_closed"]=true;
     }
     r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;pin.restore();row["affinity_restored"]=restored;row["restored_mask"]=kimi_r437_owner::mask();row["completed"]=true;save();reports[rank]=row;
    }catch(const std::exception&e){row["error"]=e.what();try{save();}catch(...){}std::fprintf(stderr,"r437 rank%d failure: %s\n",rank,e.what());std::terminate();}
   });
   for(auto&t:threads)t.join();result["all8_owners_joined_before_group_destroy"]=true;
  }
  std::vector<int>masks;for(const auto&row:reports){auto v=row.at("owner_mask").get<std::vector<int>>();masks.insert(masks.end(),v.begin(),v.end());}std::sort(masks.begin(),masks.end());if(masks.size()!=32||std::adjacent_find(masks.begin(),masks.end())!=masks.end())throw std::runtime_error("437 eight disjoint width4 owner masks");
  result["node_rank"]=node;result["reports"]=reports;result["normal_CommunicationGroup_close"]=true;result["completed"]=true;serialized=result.dump();return 0;
 }catch(const std::exception&e){result["error"]=e.what();serialized=result.dump();return 1;}
}

extern "C" const char*modern_kimi_owner_affinity_preflight_r437(){J q={{"passed",false},{"Runtime_created",false},{"checkpoint_read",false}};try{auto rows=kimi_r437_owner::check(false);q["owners"]=J::array();for(const auto&v:rows)q["owners"].push_back({{"entry_mask",v.entry},{"owner_mask",v.owner},{"restored_mask",v.after},{"restored",v.restored}});q["passed"]=true;}catch(const std::exception&e){q["error"]=e.what();}serialized=q.dump();return serialized.c_str();}
