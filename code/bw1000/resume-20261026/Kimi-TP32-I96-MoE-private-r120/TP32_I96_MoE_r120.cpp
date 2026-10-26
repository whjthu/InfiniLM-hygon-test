// Standalone HOST adapter. Existing Native305726 supplies original layer1 math.
#include "contract_r120.hpp"
#include "runtime/runtime.hpp"
#include "config/config_factory.hpp"
#include "models/kimi_k3/kimi_k3_moe.hpp"
#include "engine/distributed/communication_group.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <set>
extern "C" int modern_kimi_I192_counts_r8669y(int,uint64_t*,size_t);
extern "C" int modern_kimi_serving_counts_r4665(int,uint64_t*,size_t);
extern "C" int modern_kimi_W2_pair_counts_r9280(int,uint64_t*,size_t);
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
namespace {
using J=nlohmann::json;namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
struct Barrier{std::mutex m;std::condition_variable cv;size_t count=0,generation=0;void wait(){std::unique_lock<std::mutex>lock(m);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
thread_local std::string result;
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
void env_is(const char*n,const char*v){auto*p=std::getenv(n);if(!p||std::string(p)!=v)throw std::invalid_argument(std::string("r120 immutable flag ")+n);}
uint64_t digest(const std::vector<uint16_t>&v){uint64_t h=1469598103934665603ULL;for(auto x:v)h=(h^x)*1099511628211ULL;return h;}
double number(uint16_t x){uint32_t b=uint32_t(x)<<16;float v;std::memcpy(&v,&b,4);return v;}
void global_barrier(rt::Runtime&r,const en::RankInfo&info){auto v=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float one=1,sum=0;r.memcpyH2D(v->data(),&one,4,false);en::allreduce_(r,v,v,infinicclSum,info.comm);r.syncStream();r.memcpyD2H(&sum,v->data(),4);if(sum!=32.f)throw std::runtime_error("r120 barrier not32 physical owners");}
std::array<uint64_t,8> counters(int rank){std::array<uint64_t,8>x{};if(modern_kimi_I192_counts_r8669y(rank,x.data(),2)||modern_kimi_serving_counts_r4665(rank,x.data()+2,4)||modern_kimi_W2_pair_counts_r9280(rank,x.data()+6,2))throw std::runtime_error("r120 counter read");return x;}
J shard_info(km::KimiK3MoE&m,int globalrank){
 auto w=m.state_dict();auto required=m.state_required_keys();size_t packed=0,bytes=0;
 for(const auto&k:required){auto it=w.find(k);if(it==w.end())throw std::runtime_error("r120 missing physical state");auto t=it->second.tensor();if(!t)throw std::runtime_error("r120 empty physical state "+k);bytes+=t->nbytes();
  rt::Shape expected;
  if(k.rfind("experts.",0)==0){
   if(k.find(".w1.weight_packed")!=std::string::npos||k.find(".w3.weight_packed")!=std::string::npos)expected={96,1792};
   else if(k.find(".w1.weight_scale")!=std::string::npos||k.find(".w3.weight_scale")!=std::string::npos)expected={96,112};
   else if(k.find(".w2.weight_packed")!=std::string::npos)expected={3584,48};
   else if(k.find(".w2.weight_scale")!=std::string::npos)expected={3584,3};
   else throw std::runtime_error("r120 unknown expert tensor");
   if(t->shape()!=expected||t->dtype()!=rt::DataType::kUInt8)throw std::runtime_error("r120 actualI96 shard mismatch "+k);++packed;
  }
 }
 if(packed!=896*6||required.size()!=5384||bytes>2ULL*1024*1024*1024)throw std::runtime_error("r120 single layer keys/budget");
 return {{"required_keys",required.size()},{"expert_keys",packed},{"required_tensor_logical_bytes",bytes},{"global_I_range",{globalrank*96,(globalrank+1)*96}},{"global_W2_packed_column_range",{globalrank*48,(globalrank+1)*48}},{"global_W2_scale_column_range",{globalrank*3,(globalrank+1)*3}},{"not_allocator_peak",true}};
}
}
extern "C" const char*modern_kimi_TP32_moe_result_r120(){return result.c_str();}
extern "C" int modern_kimi_TP32_moe_r120(const char*payload){
 J d;try{
  if(!payload)throw std::invalid_argument("r120 payload");const J q=J::parse(payload);const int node=q.at("node_rank").get<int>();if(node<0||node>3)throw std::invalid_argument("r120 node0..3");
  env_is("INFINILM_KIMI_SPEC_MATH_R3676","0");env_is("INFINILM_KIMI_MOE_WORKSPACE_R4665","0");env_is("INFINILM_KIMI_MOE_EXECUTION_R4665","0");env_is("INFINILM_KIMI_I192_GROUPED_R8669Y","0");env_is("INFINILM_KIMI_BATCH_I192_R9125","0");env_is("INFINILM_KIMI_PROFILE_R9125","0");env_is("INFINILM_MXFP4_TILE_TP_R2559","0");env_is("INFINILM_OWNER_CPU_WIDTH_R2392","4");
  auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();if(cfg->get<size_t>("moe_intermediate_size")!=3072||cfg->get<size_t>("hidden_size")!=7168)throw std::invalid_argument("r120 original config");
  en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=4;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr").get<std::string>();topology.master_port=q.at("master_port").get<int>();
  std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;
  {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
   for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{
    auto info=group.get_rank_info(rank);r120::topology(node,8,info.tp_size,info.tp_rank,info.pp_size,info.pp_stage,info.world_size,info.world_rank);if(info.tp_rank!=node*8+rank||!info.comm||info.device.type()!=rt::Device::Type::kHygon||info.device.index()!=rank)throw std::runtime_error("r120 real global rank/device");
    infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"local_rank",rank},{"node_rank",node},{"global_tp_rank",info.tp_rank},{"TP",32},{"PP",1},{"arms",J::array()}};
    {auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);auto module=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});row["physical_before_load"]=shard_info(*module,info.tp_rank);
     std::unordered_map<std::string,rt::Tensor>weights;
     for(const auto&z:q.at("weights")){const int t=z.at("dtype");if(t<0||t>2)throw std::invalid_argument("r120 dtype");const auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;const auto name=z.at("name").get<std::string>();if(!weights.emplace(name,rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0})).second)throw std::invalid_argument("r120 duplicate checkpoint key");}
     auto keys=module->state_required_keys();if(keys.size()!=weights.size())throw std::runtime_error("r120 original global descriptor keycount");for(auto&k:keys)if(!weights.count(k))throw std::runtime_error("r120 missing weight "+k);
     const auto load_start=std::chrono::steady_clock::now();module->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*module);weights.clear();row["installation_inclusive_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-load_start).count();row["physical_after_load"]=shard_info(*module,info.tp_rank);
     modern_kimi_set_grouped_prefill(0);modern_kimi_set_tiled_w13_r1861(0);global_barrier(r,info);row["all32_loaded_barrier"]=true;
     for(size_t T:std::array<size_t,3>{8,16,64}){
      auto storage=rt::Tensor::empty(r,{1,T+1,7168},rt::DataType::kBFloat16);std::vector<uint16_t>original((T+1)*7168,uint16_t{0x3f80});std::memcpy(original.data(),reinterpret_cast<void*>(q.at("input_pointers").at(std::to_string(T)).get<uintptr_t>()),T*7168*2);r.memcpyH2D(storage->data(),original.data(),storage->nbytes(),false);auto input=storage->narrow({{1,0,T}});std::vector<uint16_t>reference;
      // Two warmed calls followed by four ordinary repeats; no A/B speed claim.
      for(int arm=0;arm<6;++arm){phase.wait();global_barrier(r,info);const auto before=counters(rank);const auto start=std::chrono::steady_clock::now();auto y=module->forward(r,input);r.syncStream();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();const auto after=counters(rank);if(before!=after)throw std::runtime_error("r120 I192/I384/pair path unexpectedly hit");if(y->shape()!=rt::Shape{1,T,7168}||y->dtype()!=rt::DataType::kBFloat16)throw std::runtime_error("r120 output geometry");
       std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());for(auto b:bits)if((b&0x7f80)==0x7f80)throw std::runtime_error("r120 nonfinite output");if(reference.empty())reference=bits;double sq=0,base=0,maxabs=0;size_t changed=0;for(size_t j=0;j<bits.size();++j){double a=number(reference[j]),b=number(bits[j]);sq+=(a-b)*(a-b);base+=a*a;maxabs=std::max(maxabs,std::abs(a-b));changed+=bits[j]!=reference[j];}
       std::vector<uint16_t>check(storage->numel());r.memcpyD2H(check.data(),storage->data(),storage->nbytes());if(check!=original)throw std::runtime_error("r120 input/canary modified");const auto file=q.at("raw_directory").get<std::string>()+"/globalrank"+std::to_string(info.tp_rank)+"-T"+std::to_string(T)+"-arm"+std::to_string(arm)+".bin";std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);if(!f)throw std::runtime_error("r120 persist output");
       row["arms"].push_back({{"T",T},{"arm",arm},{"warm",arm<2},{"whole_MoE_including_outer_drain_ms",ms},{"input_readonly_tail_canary",true},{"nonfinite",0},{"output_FNVword16",digest(bits)},{"repeat_changed_bits_diagnostic",changed},{"repeat_relative_RMS_diagnostic",std::sqrt(sq/std::max(base,1e-30))},{"repeat_maxabs_diagnostic",maxabs},{"private_provider_counter_delta_zero",true},{"expected_provider","SDK genericI96 impl1 source-selected"}});
      }
     }
     global_barrier(r,info);module.reset();r.syncStream();row["module_closed"]=true;
    }r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;reports[rank]=row;
   }catch(const std::exception&e){std::fprintf(stderr,"r120 fatal node=%d localrank=%d %s\n",node,rank,e.what());std::terminate();}});
   for(auto&t:threads)t.join();d["all8normal_before_communicators"]=true;
  }d["node_rank"]=node;d["reports"]=reports;d["completed"]=true;d["scope"]="standalone original layer1 TP32 genericI96; no fullModel/head/KDA";result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}
}
