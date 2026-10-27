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
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_runtime_api.h>
extern "C" int modern_kimi_prefill_pool_trim_stats_r130(int,uint64_t*,size_t);
extern "C" int modern_grouped_mxfp4_owned(infinilm::runtime::Runtime*,size_t,size_t,size_t,size_t,size_t,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*);
extern "C" int modern_grouped_plan_info(size_t*);
extern "C" int modern_kimi_W2_pair_counts_r9280(int,uint64_t*,size_t);
extern "C" int modern_kimi_I192_counts_r8669y(int,uint64_t*,size_t);
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
extern "C" int modern_kimi_serving_counts_r4665(int,uint64_t*,size_t);
namespace {using J=nlohmann::json;namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
struct Barrier{std::mutex m;std::condition_variable cv;size_t count=0,generation=0;void wait(){std::unique_lock<std::mutex>lock(m);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
thread_local std::string result;
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
struct DriverSamplerR132{
 int device;std::atomic<bool>stop{false},failed{false};std::atomic<size_t>low{SIZE_MAX},samples{0};std::thread thread;
 explicit DriverSamplerR132(int d):device(d){thread=std::thread([this]{try{infini::rt::DeviceGuard g{rt::Device::Type::kHygon,device,true};while(!stop.load()){size_t f=0,t=0;if(cudaMemGetInfo(&f,&t)!=cudaSuccess){failed=true;break;}auto p=low.load();while(f<p&&!low.compare_exchange_weak(p,f)){}++samples;std::this_thread::sleep_for(std::chrono::milliseconds(2));}}catch(...){failed=true;}});}
 void finish(){stop=true;if(thread.joinable())thread.join();if(failed||!samples)throw std::runtime_error("r132 bounded driver sampler failed/no samples");}
 ~DriverSamplerR132(){stop=true;if(thread.joinable())thread.join();}
};

}
extern "C" const char*modern_kimi_prefill_pool_result_r132(){return result.c_str();}
extern "C" int modern_kimi_prefill_pool_run_r132(const char*payload){
 J d;try{const char*on=std::getenv("KIMI_PREFILL_POOL_COMPONENT_R132");if(!on||std::string(on)!="1")throw std::invalid_argument("r132 component defaultOFF");using CPUCopyStats=int(*)(uint64_t*,size_t);auto copyStats=reinterpret_cast<CPUCopyStats>(dlsym(RTLD_DEFAULT,"modern_cpu_copy_stats_r2515"));uint64_t copyBefore[5],copyAfter[5];if(!copyStats||copyStats(copyBefore,5))throw std::runtime_error("r9280 Runtime CPUcopy stats unavailable");const J q=J::parse(payload);auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;cpu_set_t process_set;CPU_ZERO(&process_set);if(sched_getaffinity(0,sizeof(process_set),&process_set))throw std::runtime_error("r9220 affinityread");std::vector<int>process_cpus;for(int k=0;k<CPU_SETSIZE;k++)if(CPU_ISSET(k,&process_set))process_cpus.push_back(k);if(process_cpus.size()<32)throw std::runtime_error("r9220 needs32cpu for8x4owners");
 const int node=q.at("node_rank").get<int>();if(node<0||node>1)throw std::invalid_argument("TP16node0or1");en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=2;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr").get<std::string>();topology.master_port=q.at("master_port").get<int>();

 {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
 for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{cpu_set_t owner_set;CPU_ZERO(&owner_set);std::vector<int>owner_cpus;for(int j=0;j<4;j++){int cpu=process_cpus.at(rank*4+j);CPU_SET(cpu,&owner_set);owner_cpus.push_back(cpu);}auto info=group.get_rank_info(rank);if(info.tp_size!=16||info.tp_rank!=node*8+rank||!info.comm||info.pp_size!=1||info.pp_stage!=0)throw std::runtime_error("actualTP16globalRankInfo");infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"owner_CPU_affinity",owner_cpus},{"rank",rank},{"node_rank",node},{"global_tp_rank",info.tp_rank},{"actual_tp_size",info.tp_size},{"arms",J::array()},{"profile",J::array()},{"current_branch",J::array()}};
 {auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);auto module=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});
 std::unordered_map<std::string,rt::Tensor>weights;
 for(const auto&z:q.at("weights")){const std::string weight_name=z.at("name");auto sh=z.at("shape").get<rt::Shape>();if(weight_name.find(".w1.weight_packed")!=std::string::npos||weight_name.find(".w3.weight_packed")!=std::string::npos){if(sh!=rt::Shape{3072,1792})throw std::runtime_error("fullW13checkpointshape");}if(weight_name.find(".w2.weight_packed")!=std::string::npos&&sh!=rt::Shape{3584,1536})throw std::runtime_error("fullW2checkpointshape");const int t=z.at("dtype");auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;weights.emplace(z.at("name"),rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0}));}
 auto keys=module->state_required_keys();if(keys.size()!=weights.size())throw std::runtime_error("fullMoE requiredweightcount");for(const auto&k:keys)if(!weights.count(k))throw std::runtime_error("fullMoE missingweight "+k);auto installationBegin=std::chrono::steady_clock::now();module->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*module);weights.clear();row["installation_inclusive_wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-installationBegin).count();modern_kimi_set_grouped_prefill(1);modern_kimi_set_tiled_w13_r1861(0);
 for(const size_t T:std::array<size_t,1>{576}){
 auto storage=rt::Tensor::empty(r,{1,T+1,7168},rt::DataType::kBFloat16);std::vector<uint16_t> input_original((T+1)*7168,uint16_t{0x3f80});std::memcpy(input_original.data(),reinterpret_cast<void*>(q.at("input_pointers").at(std::to_string(T)).get<uintptr_t>()),T*7168*2);r.memcpyH2D(storage->data(),input_original.data(),storage->nbytes(),false);r.syncStream();auto input=storage->narrow({{1,0,T}});std::array<std::vector<uint16_t>,2>gold;

 phase.wait();if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","1",1);setenv("INFINILM_KIMI_BATCH_I192_R9125","1",1);unsetenv("INFINILM_KIMI_PROFILE_R9125");}phase.wait();
 uint64_t count_before[2],count_after[2],pair_before[2],pair_after[2];if(modern_kimi_I192_counts_r8669y(rank,count_before,2)||modern_kimi_W2_pair_counts_r9280(rank,pair_before,2))throw std::runtime_error("r132 initial nativecounts");
 Dl_info helper_info{};if(!dladdr(reinterpret_cast<void*>(&modern_grouped_mxfp4_owned),&helper_info)||!helper_info.dli_fname)throw std::runtime_error("r132 helper dladdr");row["actual_helper_path"]=helper_info.dli_fname;
 auto planinfo=reinterpret_cast<int(*)(size_t*)>(dlsym(RTLD_DEFAULT,"modern_grouped_plan_info"));if(!planinfo)throw std::runtime_error("r132 helper planinfo");
 auto verify_affinity=[&]{cpu_set_t actual;CPU_ZERO(&actual);if(sched_getaffinity(0,sizeof(actual),&actual))throw std::runtime_error("r132 affinityread");std::vector<int>got;for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&actual))got.push_back(c);if(got!=owner_cpus)throw std::runtime_error("r132 Runtime owneraffinity mismatch");};
 auto global_barrier=[&]{auto v=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float x=1,y=0;r.memcpyH2D(v->data(),&x,4,false);r.syncStream();en::allreduce_(r,v,v,infinicclSum,info.comm);r.syncStream();r.memcpyD2H(&y,v->data(),4);if(y!=16)throw std::runtime_error("r132 TP16 barrier");};
 std::vector<uint16_t>reference;auto fp=[](uint16_t x){uint32_t u=uint32_t(x)<<16;float f;std::memcpy(&f,&u,4);return double(f);};
 // WarmOFF2 then four matched OFF/ON pairs in ABBAABBA order. No common-entry
 // trim secretly normalizes away retained cache. First-live peak remains visible.
 const std::array<int,10>schedule{{0,0,0,1,1,0,0,1,1,0}};
 for(int arm=0;arm<10;++arm){
  const bool enabled=schedule[arm]!=0;phase.wait();if(rank==0){setenv("INFINILM_KIMI_PREFILL_POOL_TRIM_R130",enabled?"1":"0",1);setenv("INFINILM_KIMI_W2_PAIR_R9280","1",1);}phase.wait();global_barrier();phase.wait();verify_affinity();
  size_t before=0,total=0,after=0;if(cudaMemGetInfo(&before,&total)!=cudaSuccess)throw std::runtime_error("r132 before driverinfo");const size_t pinned_before=r.graphPinnedMemoryBytes();uint64_t stats_before[7],stats_after[7];if(modern_kimi_prefill_pool_trim_stats_r130(rank,stats_before,7))throw std::runtime_error("r132 prestats");
  DriverSamplerR132 sampler(rank);while(!sampler.samples.load()&&!sampler.failed.load())std::this_thread::yield();auto tick=std::chrono::steady_clock::now();auto y=module->forward(r,input);r.syncStream();const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();sampler.finish();if(cudaMemGetInfo(&after,&total)!=cudaSuccess)throw std::runtime_error("r132 after driverinfo");const size_t pinned_after=r.graphPinnedMemoryBytes();if(modern_kimi_prefill_pool_trim_stats_r130(rank,stats_after,7))throw std::runtime_error("r132 poststats");
  if(stats_after[0]-stats_before[0]!=(enabled?1u:0u)||stats_after[1]-stats_before[1]!=(enabled?T:0u))throw std::runtime_error("r132 actual longhelper trim coverage");size_t plan[5];if(planinfo(plan))throw std::runtime_error("r132 actualhelperplan");
  std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());for(auto v:bits)if((v&0x7f80)==0x7f80)throw std::runtime_error("r132 nonfinite");if(reference.empty())reference=bits;double d2=0,base2=0,maxabs=0;size_t diff=0;for(size_t j=0;j<bits.size();++j){auto x=fp(reference[j]),z=fp(bits[j]);auto e=z-x;d2+=e*e;base2+=x*x;maxabs=std::max(maxabs,std::abs(e));diff+=bits[j]!=reference[j];}const double nrms=std::sqrt(d2/std::max(base2,1e-30));if(nrms>0.01)throw std::runtime_error("r132 finite numeric tolerance failure");
  std::vector<uint16_t>checked(storage->numel());r.memcpyD2H(checked.data(),storage->data(),storage->nbytes());if(checked!=input_original)throw std::runtime_error("r132 inputreadonly tailcanary");
  row["arms"].push_back({{"T",T},{"arm",arm},{"warm",arm<2},{"trim_after_helper",enabled},{"entry_trim",false},{"full_MoE_complete_wall_s",wall},{"driver_free_before",before},{"driver_free_after",after},{"driver_sampled_min_free",sampler.low.load()},{"driver_samples",sampler.samples.load()},{"sampling_interval_ms",2},{"sampled_not_exact_peak",true},{"native_graph_pinned_before",pinned_before},{"native_graph_pinned_after",pinned_after},{"native_pool_allocated_cached_bytes","UNKNOWN: no public stats ABI"},{"trim_stats_before7",std::vector<uint64_t>(stats_before,stats_before+7)},{"trim_stats_after7",std::vector<uint64_t>(stats_after,stats_after+7)},{"actual_helper_plan5",std::vector<size_t>(plan,plan+5)},{"original_outer_completion",true},{"output_relative_RMS",nrms},{"output_maxabs",maxabs},{"bit_differences_diagnostic_only",diff},{"finite",true},{"input_readonly_and_tail_canary",true},{"sampling_perturbs_botharms",true}});
 }
 if(modern_kimi_I192_counts_r8669y(rank,count_after,2)||modern_kimi_W2_pair_counts_r9280(rank,pair_after,2))throw std::runtime_error("r132 finalcounts");if(count_after[0]!=count_before[0]||count_after[1]!=count_before[1]||pair_after[0]!=pair_before[0]||pair_after[1]!=pair_before[1])throw std::runtime_error("r132 T576 must not call T<=64 I192/W2pair");row["current_branch"].push_back({{"T",T},{"I192_calls",0},{"W2pair_calls",0},{"actual_longhelper_trim_ONcalls",4},{"scope","singleL1 TP16 constructedrepeat input, not real478activation or fullModel/PP"}});
 }phase.wait();if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);setenv("INFINILM_KIMI_W2_PAIR_R9280","0",1);setenv("INFINILM_KIMI_PREFILL_POOL_TRIM_R130","0",1);}phase.wait();module.reset();r.syncStream();row["module_closed"]=true;}r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;reports[rank]=row;
 }catch(const std::exception&e){std::fprintf(stderr,"caller8669o rank%d %s\n",rank,e.what());std::terminate();}});
 for(auto&t:threads)t.join();d["all8normal_before_communicators"]=true;}if(copyStats(copyAfter,5))throw std::runtime_error("r9280 CPUcopy read after");J copyDelta=J::array();for(size_t i=0;i<5;++i){if(copyAfter[i]<copyBefore[i])throw std::runtime_error("r9280 CPUcopy nonmonotonic");copyDelta.push_back(copyAfter[i]-copyBefore[i]);}d["CPUcopy_global_process_delta"]=copyDelta;d["CPUcopy_scope"]="entire native session including original weightsinstallation, forward and teardown; pernode aggregate not perowner or isolatedGPU-H2D timing";d["node_rank"]=node;d["expected_native_expert_shard"]={{"W13",{896,384,1792}},{"S13",{896,384,112}},{"W2",{896,3584,96}},{"S2",{896,3584,6}}};d["reports"]=reports;d["completed"]=true;result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}}
