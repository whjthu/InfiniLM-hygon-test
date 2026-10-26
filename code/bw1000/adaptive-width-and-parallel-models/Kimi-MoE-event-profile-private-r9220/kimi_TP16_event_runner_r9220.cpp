#include "runtime/runtime.hpp"
#include "config/config_factory.hpp"
#include "models/kimi_k3/kimi_k3_moe.hpp"
#include "engine/distributed/communication_group.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <array>
#include <thread>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
extern "C" int modern_kimi_moe_event_arm_r9220(void*,int,uint64_t);
extern "C" const char*modern_kimi_moe_event_read_r9220();
extern "C" int modern_kimi_moe_event_close_r9220();
extern "C" int modern_kimi_I192_counts_r8669y(int,uint64_t*,size_t);
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
extern "C" int modern_kimi_serving_counts_r4665(int,uint64_t*,size_t);
namespace {using J=nlohmann::json;namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
struct Barrier{std::mutex m;std::condition_variable cv;size_t count=0,generation=0;void wait(){std::unique_lock<std::mutex>lock(m);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
thread_local std::string result;
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
using Begin=void(*)(int,int);using End=const char*(*)();
}
extern "C" const char*modern_kimi_TP16_moe_result_r8669w(){return result.c_str();}
extern "C" int modern_kimi_TP16_moe_r8669w(const char*payload){
 J d;try{const J q=J::parse(payload);auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;
 const int node=q.at("node_rank").get<int>();if(node<0||node>1)throw std::invalid_argument("TP16node0or1");en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=2;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr").get<std::string>();topology.master_port=q.at("master_port").get<int>();

 {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
 for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{auto info=group.get_rank_info(rank);if(info.tp_size!=16||info.tp_rank!=node*8+rank||!info.comm||info.pp_size!=1||info.pp_stage!=0)throw std::runtime_error("actualTP16globalRankInfo");infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"rank",rank},{"node_rank",node},{"global_tp_rank",info.tp_rank},{"actual_tp_size",info.tp_size},{"arms",J::array()}};
 {auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);auto module=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});
 std::unordered_map<std::string,rt::Tensor>weights;
 for(const auto&z:q.at("weights")){const std::string weight_name=z.at("name");auto sh=z.at("shape").get<rt::Shape>();if(weight_name.find(".w1.weight_packed")!=std::string::npos||weight_name.find(".w3.weight_packed")!=std::string::npos){if(sh!=rt::Shape{3072,1792})throw std::runtime_error("fullW13checkpointshape");}if(weight_name.find(".w2.weight_packed")!=std::string::npos&&sh!=rt::Shape{3584,1536})throw std::runtime_error("fullW2checkpointshape");const int t=z.at("dtype");auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;weights.emplace(z.at("name"),rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0}));}
 auto keys=module->state_required_keys();if(keys.size()!=weights.size())throw std::runtime_error("fullMoE requiredweightcount");for(const auto&k:keys)if(!weights.count(k))throw std::runtime_error("fullMoE missingweight "+k);module->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*module);weights.clear();modern_kimi_set_grouped_prefill(0);modern_kimi_set_tiled_w13_r1861(1);
 for(const size_t T:std::array<size_t,4>{8,16,32,64}){
 auto storage=rt::Tensor::empty(r,{1,T+1,7168},rt::DataType::kBFloat16);std::vector<uint16_t> input_original((T+1)*7168,uint16_t{0x3f80});std::memcpy(input_original.data(),reinterpret_cast<void*>(q.at("input_pointers").at(std::to_string(T)).get<uintptr_t>()),T*7168*2);r.memcpyH2D(storage->data(),input_original.data(),storage->nbytes(),false);r.syncStream();auto input=storage->narrow({{1,0,T}});std::array<std::vector<uint16_t>,2>gold;

 phase.wait();if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","1",1);setenv("INFINILM_KIMI_BATCH_I192_R9125","1",1);unsetenv("INFINILM_KIMI_PROFILE_R9125");}phase.wait();
 for(int arm=0;arm<4;arm++){
  // One warm, OFF bracket, armed once, OFF bracket. Original completion is unchanged.
  phase.wait();auto scalar=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float one=1.f,sum=0.f;r.memcpyH2D(scalar->data(),&one,4,false);r.syncStream();en::allreduce_(r,scalar,scalar,infinicclSum,info.comm);r.syncStream();r.memcpyD2H(&sum,scalar->data(),4);if(sum!=16.f)throw std::runtime_error("r9220 global16barrier");
  if(arm==2&&modern_kimi_moe_event_arm_r9220(&r,1,uint64_t(T)))throw std::runtime_error("r9220 arm");
  auto begin=std::chrono::steady_clock::now();auto y=module->forward(r,input);r.syncStream();double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  if(arm==2){auto payload=modern_kimi_moe_event_read_r9220();if(!payload)throw std::runtime_error("r9220 read after originalcompletion");row["profile"].push_back(J::parse(payload));}
  std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());for(auto b:bits)if((b&0x7f80)==0x7f80)throw std::runtime_error("r9220 nonfinite");std::vector<uint16_t>checked(storage->numel());r.memcpyD2H(checked.data(),storage->data(),storage->nbytes());if(checked!=input_original)throw std::runtime_error("r9220 readonly/canary");
  row["arms"].push_back({{"T",T},{"arm",arm},{"kind",arm==0?"warm":arm==2?"armed_once":"profile_OFF_bracket"},{"full_MoE_complete_wall_ms",wall},{"input_readonly_and_tail_canary",true},{"original_outer_drain_only",true}});
 }
 }phase.wait();if(rank==0)setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);phase.wait();if(modern_kimi_moe_event_close_r9220())throw std::runtime_error("r9220 ownerclose beforeRuntime");module.reset();r.syncStream();row["module_closed"]=true;}r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;reports[rank]=row;
 }catch(const std::exception&e){std::fprintf(stderr,"caller8669o rank%d %s\n",rank,e.what());std::terminate();}});
 for(auto&t:threads)t.join();d["all8normal_before_communicators"]=true;}d["node_rank"]=node;d["expected_native_expert_shard"]={{"W13",{896,384,1792}},{"S13",{896,384,112}},{"W2",{896,3584,96}},{"S2",{896,3584,6}}};d["reports"]=reports;d["completed"]=true;result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}}
