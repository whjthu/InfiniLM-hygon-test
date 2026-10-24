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
extern "C" int modern_kimi_I192_counts_r8669y(int,uint64_t*,size_t);
extern "C" int modern_kimi_EP2_counts_r8669zj(int,uint64_t*,size_t);
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
extern "C" int modern_kimi_serving_counts_r4665(int,uint64_t*,size_t);
namespace {using J=nlohmann::json;namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
struct Barrier{std::mutex m;std::condition_variable cv;size_t count=0,generation=0;void wait(){std::unique_lock<std::mutex>lock(m);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
thread_local std::string result;
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
using Begin=void(*)(int,int);using End=const char*(*)();
}
extern "C" const char*modern_kimi_TP16_moe_result_r8669zl(){return result.c_str();}
extern "C" int modern_kimi_TP16_moe_r8669zl(const char*payload){
 J d;try{const J q=J::parse(payload);auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;
 if(!q.at("activity_profile").get<bool>())throw std::runtime_error("finite_activity_only");
 using ProfilerControl=int(*)();auto profiler_start=reinterpret_cast<ProfilerControl>(dlsym(RTLD_DEFAULT,"hipProfilerStart"));auto profiler_stop=reinterpret_cast<ProfilerControl>(dlsym(RTLD_DEFAULT,"hipProfilerStop"));
 if(!profiler_start||!profiler_stop)throw std::runtime_error("existing_hipprof_StartStop_unavailable");
 auto epoch_ns=[](){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();};
 const int node=q.at("node_rank").get<int>();if(node<0||node>1)throw std::invalid_argument("TP16node0or1");en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=2;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr").get<std::string>();topology.master_port=q.at("master_port").get<int>();

 {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
 for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{auto info=group.get_rank_info(rank);if(info.tp_size!=16||info.tp_rank!=node*8+rank||!info.comm||info.pp_size!=1||info.pp_stage!=0)throw std::runtime_error("actualTP16globalRankInfo");infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"rank",rank},{"node_rank",node},{"global_tp_rank",info.tp_rank},{"actual_tp_size",info.tp_size},{"arms",J::array()}};
 {std::array<std::unique_ptr<km::KimiK3MoE>,2>modules;
 for(int kind=0;kind<2;++kind){auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);infinilm::layers::moe::EPConfig ep;if(kind){ep.backend=infinilm::layers::moe::EPBackend::LocalAllReduce;ep.ep_size=2;ep.ep_rank=node;}modules[kind]=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,ep);}

 std::unordered_map<std::string,rt::Tensor>weights;
 for(const auto&z:q.at("weights")){const std::string weight_name=z.at("name");auto sh=z.at("shape").get<rt::Shape>();if(weight_name.find(".w1.weight_packed")!=std::string::npos||weight_name.find(".w3.weight_packed")!=std::string::npos){if(sh!=rt::Shape{3072,1792})throw std::runtime_error("fullW13checkpointshape");}if(weight_name.find(".w2.weight_packed")!=std::string::npos&&sh!=rt::Shape{3584,1536})throw std::runtime_error("fullW2checkpointshape");const int t=z.at("dtype");auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;weights.emplace(z.at("name"),rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0}));}
 for(int kind=0;kind<2;++kind){auto&module=modules[kind];auto keys=module->state_required_keys();std::unordered_map<std::string,rt::Tensor>selected;
 size_t expert_keys=0;for(const auto&k:keys){if(!weights.count(k))throw std::runtime_error("EP2 missingFQN "+k);selected.emplace(k,weights.at(k));if(k.find("experts.")==0)++expert_keys;}
 row["weight_binding"].push_back({{"kind",kind},{"keys",keys},{"expert_keys",expert_keys},{"global_expert_start",kind?node*448:0},{"local_experts",kind?448:896},{"expert_TP_rank",kind?info.tp_rank%8:info.tp_rank},{"expert_TP_size",kind?8:16},{"true_communicator_TP_size",16}});
 module->load_parameters_no_sync(r,selected,true);r.syncStream();post(r,*module);}
 weights.clear();modern_kimi_set_grouped_prefill(0);modern_kimi_set_tiled_w13_r1861(1);

 auto input=rt::Tensor::empty(r,{1,16,7168},rt::DataType::kBFloat16);r.memcpyH2D(input->data(),reinterpret_cast<void*>(q.at("input_pointer").get<uintptr_t>()),input->nbytes(),false);r.syncStream();std::array<std::vector<uint16_t>,2>gold;
 const std::array<int,5>plan={0,1,0,0,1};
 for(int arm=0;arm<int(plan.size());++arm){phase.wait();if(rank==0)setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","1",1);phase.wait();
 std::fprintf(stderr,"TP16_PHASE node=%d globalrank=%d arm=%d BARRIER_BEGIN\n",node,info.tp_rank,arm);
 auto barrier_start=std::chrono::steady_clock::now();auto scalar=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float one=1.f,sum=0.f;r.memcpyH2D(scalar->data(),&one,sizeof(one),false);r.syncStream();en::allreduce_(r,scalar,scalar,infinicclSum,info.comm);r.syncStream();r.memcpyD2H(&sum,scalar->data(),sizeof(sum));if(sum!=16.f)throw std::runtime_error("TP16globalbarrier_sum_not16");
 row["barriers"].push_back({{"arm",arm},{"participants_sum",sum},{"outside_timing",true},{"wall_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-barrier_start).count()}});
 std::fprintf(stderr,"TP16_PHASE node=%d globalrank=%d arm=%d BARRIER_DONE sum=%g\n",node,info.tp_rank,arm,sum);
uint64_t ep_before[3],ep_after[3];if(modern_kimi_EP2_counts_r8669zj(rank,ep_before,3))throw std::runtime_error("EP2counterbefore");
uint64_t custom_before[2],custom_after[2];if(modern_kimi_I192_counts_r8669y(rank,custom_before,2))throw std::runtime_error("I192counter_before");
uint64_t before[4],after[4];if(modern_kimi_serving_counts_r4665(rank,before,4))throw std::runtime_error("servingcounter");
 const bool trace=arm>=3;
 // Existing process-global profiler is toggled by exactly one owner, with CPU
 // rendezvous only outside original forward. No new native/stream sync added.
 phase.wait();if(trace&&rank==0){auto begin=epoch_ns();int rc=profiler_start();if(rc)throw std::runtime_error("hipProfilerStart_failed");row["profile_windows"].push_back({{"arm",arm},{"EP2",bool(plan[arm])},{"begin_realtime_ns",begin},{"after_start_realtime_ns",epoch_ns()},{"start_rc",rc},{"scope","one process/all8devices; node-local epoch only; not across-host subtraction"}});}phase.wait();
 const auto start=std::chrono::steady_clock::now();auto y=modules[plan[arm]]->forward(r,input);r.syncStream();double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 phase.wait();if(trace&&rank==0){auto end=epoch_ns();int rc=profiler_stop();auto&window=row["profile_windows"].back();window["before_stop_realtime_ns"]=end;window["end_realtime_ns"]=epoch_ns();window["stop_rc"]=rc;if(rc)throw std::runtime_error("hipProfilerStop_failed");}phase.wait();
 if(modern_kimi_serving_counts_r4665(rank,after,4))throw std::runtime_error("servingcounter");std::array<uint64_t,4>delta{};for(int i=0;i<4;++i)delta[i]=after[i]-before[i];if(delta!=std::array<uint64_t,4>{0,0,0,0})throw std::runtime_error("I192mustuseactualfallbacknotvariant4");std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());size_t nf=0;for(auto b:bits)nf+=(b&0x7f80)==0x7f80;if(nf)throw std::runtime_error("nonfinite");if(arm<2)gold[plan[arm]]=bits;bool exact=bits==gold[plan[arm]];
if(modern_kimi_I192_counts_r8669y(rank,custom_after,2))throw std::runtime_error("I192counter_after");std::array<uint64_t,2>cd{};for(int j=0;j<2;++j)cd[j]=custom_after[j]-custom_before[j];if(cd!=std::array<uint64_t,2>{uint64_t(!plan[arm]),uint64_t(!plan[arm]*16)})throw std::runtime_error("I192actual_hit_mismatch");
 if(modern_kimi_EP2_counts_r8669zj(rank,ep_after,3)||ep_after[0]-ep_before[0]!=uint64_t(plan[arm])||ep_after[1]-ep_before[1]!=uint64_t(plan[arm]*16)||(plan[arm]&&ep_after[2]!=uint64_t(node*448)))throw std::runtime_error("EP2actualhit");
 row["arms"].push_back({{"arm",arm},{"kind",arm<2?"warm":arm==2?"original_anchor_traceOFF":"instrumented_diagnostic"},{"candidate_ON",bool(plan[arm])},{"I192_counter_delta",cd},{"EP2_counter_after",std::array<uint64_t,3>{ep_after[0],ep_after[1],ep_after[2]}},{"full_MoE_complete_wall_ms",wall},{"outer_drain_included",true},{"actual_provider_delta",delta},{"ownrepeat_exact",exact}});
 // Persist actual outputs before classification; no intermediatebit Model gate.
 std::string file=q.at("raw_directory").get<std::string>()+"/rank"+std::to_string(rank)+"-arm"+std::to_string(arm)+".bin";std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);if(!f)throw std::runtime_error("outputpersist");if(!exact)row["numerical_repeat_mismatch"]=true;
 }phase.wait();if(rank==0)setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);phase.wait();for(auto&m:modules)m.reset();r.syncStream();row["module_closed"]=true;}r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;reports[rank]=row;
 }catch(const std::exception&e){std::fprintf(stderr,"caller8669o rank%d %s\n",rank,e.what());std::terminate();}});
 for(auto&t:threads)t.join();d["all8normal_before_communicators"]=true;}d["node_rank"]=node;d["expected_native_expert_shard"]={{"W13",{896,384,1792}},{"S13",{896,384,112}},{"W2",{896,3584,96}},{"S2",{896,3584,6}}};d["reports"]=reports;d["completed"]=true;result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}}
