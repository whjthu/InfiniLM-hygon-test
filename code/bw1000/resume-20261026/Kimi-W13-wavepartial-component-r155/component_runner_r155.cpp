#include "runtime/runtime.hpp"
#include "config/config_factory.hpp"
#include "models/kimi_k3/kimi_k3_moe.hpp"
#include "engine/distributed/communication_group.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <array>
#include <thread>
#include <sched.h>
#include <pthread.h>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
extern "C" int modern_kimi_W13_wavepartial_counts_r155(int,uint64_t*,size_t);
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
 J d;try{const J q=J::parse(payload);const int layer=q.at("layer").get<int>();if(layer!=1&&layer!=47)throw std::invalid_argument("r155 L1/L47 only");auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;cpu_set_t process_set;CPU_ZERO(&process_set);if(sched_getaffinity(0,sizeof(process_set),&process_set))throw std::runtime_error("r9220 affinityread");std::vector<int>process_cpus;for(int k=0;k<CPU_SETSIZE;k++)if(CPU_ISSET(k,&process_set))process_cpus.push_back(k);if(process_cpus.size()<32)throw std::runtime_error("r9220 needs32cpu for8x4owners");
 const int node=q.at("node_rank").get<int>();if(node<0||node>1)throw std::invalid_argument("TP16node0or1");en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=2;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr").get<std::string>();topology.master_port=q.at("master_port").get<int>();

 {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
 for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{cpu_set_t owner_set;CPU_ZERO(&owner_set);std::vector<int>owner_cpus;for(int j=0;j<4;j++){int cpu=process_cpus.at(rank*4+j);CPU_SET(cpu,&owner_set);owner_cpus.push_back(cpu);}auto info=group.get_rank_info(rank);if(info.tp_size!=16||info.tp_rank!=node*8+rank||!info.comm||info.pp_size!=1||info.pp_stage!=0)throw std::runtime_error("actualTP16globalRankInfo");infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"owner_CPU_affinity",owner_cpus},{"rank",rank},{"node_rank",node},{"global_tp_rank",info.tp_rank},{"actual_tp_size",info.tp_size},{"arms",J::array()}};
 {auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);auto module=std::make_unique<km::KimiK3MoE>(r,config,layer,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});
 std::unordered_map<std::string,rt::Tensor>weights;
 for(const auto&z:q.at("weights")){const std::string weight_name=z.at("name");auto sh=z.at("shape").get<rt::Shape>();if(weight_name.find(".w1.weight_packed")!=std::string::npos||weight_name.find(".w3.weight_packed")!=std::string::npos){if(sh!=rt::Shape{3072,1792})throw std::runtime_error("fullW13checkpointshape");}if(weight_name.find(".w2.weight_packed")!=std::string::npos&&sh!=rt::Shape{3584,1536})throw std::runtime_error("fullW2checkpointshape");const int t=z.at("dtype");auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;weights.emplace(z.at("name"),rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0}));}
 auto keys=module->state_required_keys();if(keys.size()!=weights.size())throw std::runtime_error("fullMoE requiredweightcount");for(const auto&k:keys)if(!weights.count(k))throw std::runtime_error("fullMoE missingweight "+k);module->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*module);weights.clear();modern_kimi_set_grouped_prefill(0);modern_kimi_set_tiled_w13_r1861(1);
 for(size_t workload=0;workload<2;++workload){const size_t T=64;
 auto storage=rt::Tensor::empty(r,{1,T+1,7168},rt::DataType::kBFloat16);std::vector<uint16_t> input_original((T+1)*7168,uint16_t{0x3f80});std::memcpy(input_original.data(),reinterpret_cast<void*>(q.at("input_pointers").at(std::to_string(workload)).get<uintptr_t>()),T*7168*2);r.memcpyH2D(storage->data(),input_original.data(),storage->nbytes(),false);r.syncStream();auto input=storage->narrow({{1,0,T}});std::array<std::vector<uint16_t>,2>gold;

 phase.wait();if(rank==0){setenv("INFINILM_KIMI_W2_PAIR_R9280","1",1);setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","1",1);setenv("INFINILM_KIMI_BATCH_I192_R9125","1",1);unsetenv("INFINILM_KIMI_PROFILE_R9125");}phase.wait();
 std::vector<uint16_t> reference;
 for(int arm=0;arm<13;arm++){const bool profiled=arm>=4;const bool candidate=(arm>=4&&(arm-4)%3==1);phase.wait();if(rank==0)setenv("INFINILM_KIMI_W13_WAVEPARTIAL_R155",candidate?"1":"0",1);phase.wait();uint64_t branch_before[2],branch_after[2];if(modern_kimi_W13_wavepartial_counts_r155(rank,branch_before,2))throw std::runtime_error("r155 branch before");J uninstrumented=J::array();if(arm>=4)for(int rep=0;rep<3;++rep){auto start=std::chrono::steady_clock::now();auto z=module->forward(r,input);r.syncStream();uninstrumented.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
  // One warm, OFF bracket, armed once, OFF bracket. Original completion is unchanged.
  phase.wait();auto scalar=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float one=1.f,sum=0.f;r.memcpyH2D(scalar->data(),&one,4,false);r.syncStream();en::allreduce_(r,scalar,scalar,infinicclSum,info.comm);r.syncStream();r.memcpyD2H(&sum,scalar->data(),4);if(sum!=16.f)throw std::runtime_error("r9220 global16barrier");
  if(profiled&&modern_kimi_moe_event_arm_r9220(&r,layer,uint64_t(layer*100+workload*1000+arm+1)))throw std::runtime_error("r9220 arm");
  auto begin=std::chrono::steady_clock::now();auto y=module->forward(r,input);auto submitted=std::chrono::steady_clock::now();r.syncStream();auto drained=std::chrono::steady_clock::now();double wall=std::chrono::duration<double,std::milli>(drained-begin).count();double tail=std::chrono::duration<double,std::milli>(drained-submitted).count();cpu_set_t actual_owner_set;CPU_ZERO(&actual_owner_set);if(sched_getaffinity(0,sizeof(actual_owner_set),&actual_owner_set))throw std::runtime_error("r9220 read actualowner affinity");std::vector<int>actual_owner_cpus;for(int c=0;c<CPU_SETSIZE;c++)if(CPU_ISSET(c,&actual_owner_set))actual_owner_cpus.push_back(c);if(actual_owner_cpus!=owner_cpus||actual_owner_cpus.size()!=4)throw std::runtime_error("r9220 afterfirstsubmit actual="+J(actual_owner_cpus).dump()+" expected="+J(owner_cpus).dump());row["owner_CPU_affinity"]=actual_owner_cpus;
  if(profiled){auto payload=modern_kimi_moe_event_read_r9220();if(!payload)throw std::runtime_error("r9220 read after originalcompletion");auto profile=J::parse(payload);profile["workload_index"]=workload;profile["arm"]=arm;profile["candidate_wavepartial"]=candidate;row["profile"].push_back(profile);}
  std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());for(auto b:bits)if((b&0x7f80)==0x7f80)throw std::runtime_error("r9220 nonfinite");if(reference.empty())reference=bits;auto fp=[](uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return double(f);};double d2=0,b2=0,mx=0;size_t different=0;for(size_t j=0;j<bits.size();j++){double a=fp(reference[j]),b=fp(bits[j]),e=b-a;d2+=e*e;b2+=a*a;mx=std::max(mx,std::abs(e));different+=bits[j]!=reference[j];}double nrmse=std::sqrt(d2/std::max(b2,1e-30));if(nrmse>0.005)throw std::runtime_error("r155 sameinput OFF/ON NRMSE exceeded diagnostic bound");
  std::vector<uint16_t>checked(storage->numel());r.memcpyD2H(checked.data(),storage->data(),storage->nbytes());if(checked!=input_original)throw std::runtime_error("r9220 readonly/canary");
  if(modern_kimi_W13_wavepartial_counts_r155(rank,branch_after,2))throw std::runtime_error("r155 branch after");for(int c=0;c<2;++c)if(branch_after[c]-branch_before[c]!=uint64_t(c==int(candidate)?(arm>=4?4:1):0))throw std::runtime_error("r155 actual branch coverage");
  row["arms"].push_back({{"candidate_wavepartial",candidate},{"uninstrumented_full_MoE_complete_wall_ms",uninstrumented},{"T",T},{"workload_index",workload},{"original_tail_drain_ms",tail},{"arm",arm},{"layer",layer},{"forward_finite",true},{"NRMSE_vs_first_warm",nrmse},{"maxabs_vs_first_warm",mx},{"bit_differences_diagnostic_only",different},{"kind",arm<4?"warm":profiled?"armed_once":"profile_OFF_bracket"},{"full_MoE_complete_wall_ms",wall},{"input_readonly_and_tail_canary",true},{"original_outer_drain_only",true}});
 }
 }phase.wait();if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);setenv("INFINILM_KIMI_W13_WAVEPARTIAL_R155","0",1);}phase.wait();if(modern_kimi_moe_event_close_r9220())throw std::runtime_error("r9220 ownerclose beforeRuntime");module.reset();r.syncStream();row["module_closed"]=true;}r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;reports[rank]=row;
 }catch(const std::exception&e){std::fprintf(stderr,"caller8669o rank%d %s\n",rank,e.what());std::terminate();}});
 for(auto&t:threads)t.join();d["all8normal_before_communicators"]=true;}d["node_rank"]=node;d["expected_native_expert_shard"]={{"W13",{896,384,1792}},{"S13",{896,384,112}},{"W2",{896,3584,96}},{"S2",{896,3584,6}}};d["reports"]=reports;d["completed"]=true;result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}}
