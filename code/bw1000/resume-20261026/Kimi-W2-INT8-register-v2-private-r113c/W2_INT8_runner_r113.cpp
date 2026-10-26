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
extern "C" const char*modern_kimi_moe_profile_r9125(int);
extern "C" int modern_kimi_W2_pair_counts_r9280(int,uint64_t*,size_t);
extern "C" int modern_kimi_I192_counts_r8669y(int,uint64_t*,size_t);
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
extern "C" int modern_kimi_serving_counts_r4665(int,uint64_t*,size_t);
extern "C" int modern_kimi_i8_layout_r113(const void*,const void*,void*,void*);
extern "C" size_t modern_kimi_i8_bytes_r113(size_t);
extern "C" int modern_kimi_i8_bind_r113(int,size_t,void*,size_t,void*);
extern "C" int modern_kimi_i8_counts_r113(int,uint64_t*,size_t);
namespace {using J=nlohmann::json;namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
struct Barrier{std::mutex m;std::condition_variable cv;size_t count=0,generation=0;void wait(){std::unique_lock<std::mutex>lock(m);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
thread_local std::string result;
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
using Begin=void(*)(int,int);using End=const char*(*)();
}
extern "C" const char*modern_kimi_W2_INT8_result_r113(){return result.c_str();}
extern "C" int modern_kimi_W2_INT8_run_r113(const char*payload){
 J d;try{using CPUCopyStats=int(*)(uint64_t*,size_t);auto copyStats=reinterpret_cast<CPUCopyStats>(dlsym(RTLD_DEFAULT,"modern_cpu_copy_stats_r2515"));uint64_t copyBefore[5],copyAfter[5];if(!copyStats||copyStats(copyBefore,5))throw std::runtime_error("r9280 Runtime CPUcopy stats unavailable");const J q=J::parse(payload);auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;cpu_set_t process_set;CPU_ZERO(&process_set);if(sched_getaffinity(0,sizeof(process_set),&process_set))throw std::runtime_error("r9220 affinityread");std::vector<int>process_cpus;for(int k=0;k<CPU_SETSIZE;k++)if(CPU_ISSET(k,&process_set))process_cpus.push_back(k);if(process_cpus.size()<32)throw std::runtime_error("r9220 needs32cpu for8x4owners");
 const int node=q.at("node_rank").get<int>();if(node<0||node>1)throw std::invalid_argument("TP16node0or1");en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=2;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr").get<std::string>();topology.master_port=q.at("master_port").get<int>();

 {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
 for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{cpu_set_t owner_set;CPU_ZERO(&owner_set);std::vector<int>owner_cpus;for(int j=0;j<4;j++){int cpu=process_cpus.at(rank*4+j);CPU_SET(cpu,&owner_set);owner_cpus.push_back(cpu);}auto info=group.get_rank_info(rank);if(info.tp_size!=16||info.tp_rank!=node*8+rank||!info.comm||info.pp_size!=1||info.pp_stage!=0)throw std::runtime_error("actualTP16globalRankInfo");infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"owner_CPU_affinity",owner_cpus},{"rank",rank},{"node_rank",node},{"global_tp_rank",info.tp_rank},{"actual_tp_size",info.tp_size},{"arms",J::array()},{"profile",J::array()},{"current_branch",J::array()}};
 {auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);auto module=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});
 std::unordered_map<std::string,rt::Tensor>weights;
 for(const auto&z:q.at("weights")){const std::string weight_name=z.at("name");auto sh=z.at("shape").get<rt::Shape>();if(weight_name.find(".w1.weight_packed")!=std::string::npos||weight_name.find(".w3.weight_packed")!=std::string::npos){if(sh!=rt::Shape{3072,1792})throw std::runtime_error("fullW13checkpointshape");}if(weight_name.find(".w2.weight_packed")!=std::string::npos&&sh!=rt::Shape{3584,1536})throw std::runtime_error("fullW2checkpointshape");const int t=z.at("dtype");auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;weights.emplace(z.at("name"),rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0}));}
 auto keys=module->state_required_keys();if(keys.size()!=weights.size())throw std::runtime_error("fullMoE requiredweightcount");for(const auto&k:keys)if(!weights.count(k))throw std::runtime_error("fullMoE missingweight "+k);auto installationBegin=std::chrono::steady_clock::now();module->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*module);weights.clear();row["installation_inclusive_wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-installationBegin).count();modern_kimi_set_grouped_prefill(1);modern_kimi_set_tiled_w13_r1861(0);
 for(int scenario=0;scenario<3;++scenario){const size_t T=scenario==0?8:64;const bool constructed=scenario==2;
 const size_t scratchbytes=modern_kimi_i8_bytes_r113(T);auto i8storage=rt::Tensor::empty(r,{scratchbytes+512},rt::DataType::kUInt8);std::vector<uint8_t> init(scratchbytes+512,0xa5);r.memcpyH2D(i8storage->data(),init.data(),init.size(),false);r.syncStream();auto*scratch=reinterpret_cast<unsigned char*>(i8storage->data())+256;if(modern_kimi_i8_bind_r113(rank,T,scratch,scratchbytes,r.stream()))throw std::runtime_error("i8scratch bind");
 struct Unbind{rt::Runtime&r;int rank;~Unbind(){try{r.syncStream();}catch(...){std::terminate();}modern_kimi_i8_bind_r113(rank,0,nullptr,0,nullptr);}}unbind{r,rank};
 if(scenario==0){auto la=rt::Tensor::empty(r,{512},rt::DataType::kUInt8),lb=rt::Tensor::empty(r,{512},rt::DataType::kUInt8),lc=rt::Tensor::empty(r,{256},rt::DataType::kInt32);std::array<int8_t,512>a{},b{};std::array<int32_t,256>expected{},actual{};for(int i=0;i<512;++i){a[i]=int8_t((i*7)%17-8);b[i]=int8_t((i*13)%19-9);}for(int m=0;m<16;++m)for(int n=0;n<16;++n)for(int k=0;k<32;++k)expected[m*16+n]+=int32_t(a[m*32+k])*int32_t(b[n*32+k]);r.memcpyH2D(la->data(),a.data(),512,false);r.memcpyH2D(lb->data(),b.data(),512,false);if(modern_kimi_i8_layout_r113(la->data(),lb->data(),lc->data(),r.stream()))throw std::runtime_error("i8layout launch");r.syncStream();r.memcpyD2H(actual.data(),lc->data(),1024);if(actual!=expected)throw std::runtime_error("i8layout CPUinteger oracle failed");row["INT8_atom_CPU_integer_oracle_all256_pass"]=true;}
 auto storage=rt::Tensor::empty(r,{1,T+1,7168},rt::DataType::kBFloat16);std::vector<uint16_t> input_original((T+1)*7168,uint16_t{0x3f80});std::memcpy(input_original.data(),reinterpret_cast<void*>(q.at("input_pointers").at(std::to_string(T)).get<uintptr_t>()),T*7168*2);r.memcpyH2D(storage->data(),input_original.data(),storage->nbytes(),false);r.syncStream();auto input=storage->narrow({{1,0,T}});std::array<std::vector<uint16_t>,2>gold;

 phase.wait();if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","1",1);setenv("INFINILM_KIMI_BATCH_I192_R9125","1",1);unsetenv("INFINILM_KIMI_PROFILE_R9125");setenv("INFINILM_KIMI_W2_PAIR_R9280","1",1);setenv("INFINILM_KIMI_W2_CONSTRUCTED_R113",constructed?"1":"0",1);}phase.wait();
 uint64_t i8before[3],i8after[3];if(modern_kimi_i8_counts_r113(rank,i8before,3))throw std::runtime_error("i8 counts before");uint64_t count_before[2],count_after[2],legacy_before[4],legacy_after[4],pair_before[2],pair_after[2];if(modern_kimi_W2_pair_counts_r9280(rank,pair_before,2)||modern_kimi_I192_counts_r8669y(rank,count_before,2)||modern_kimi_serving_counts_r4665(rank,legacy_before,4))throw std::runtime_error("r9264 initial counters");
 auto verify_affinity=[&]{cpu_set_t actual;CPU_ZERO(&actual);if(sched_getaffinity(0,sizeof(actual),&actual))throw std::runtime_error("r9264 affinity read");std::vector<int>got;for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&actual))got.push_back(c);if(got!=owner_cpus)throw std::runtime_error("r9264 original Runtime owner autoaffinity mismatch");};
 auto global_barrier=[&]{auto v=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float x=1,y=0;r.memcpyH2D(v->data(),&x,4,false);r.syncStream();en::allreduce_(r,v,v,infinicclSum,info.comm);r.syncStream();r.memcpyD2H(&y,v->data(),4);if(y!=16)throw std::runtime_error("r9264 real TP16 barrier");};
 for(int enabled:{0,1}){phase.wait();if(rank==0)setenv("INFINILM_KIMI_W2_INT8_R113",enabled?"1":"0",1);phase.wait();for(int w=0;w<2;++w){auto y=module->forward(r,input);r.syncStream();}}verify_affinity();
 std::vector<uint16_t>reference;auto fp=[](uint16_t x){uint32_t u=uint32_t(x)<<16;float f;std::memcpy(&f,&u,4);return double(f);};
 for(int arm=0;arm<8;++arm){
  const bool enabled=(arm==1||arm==2||arm==5||arm==6);phase.wait();if(rank==0)setenv("INFINILM_KIMI_W2_INT8_R113",enabled?"1":"0",1);phase.wait();global_barrier();phase.wait();
  phase.wait();J samples=J::array();rt::Tensor y;
  for(int rep=0;rep<8;++rep){
   const auto realtime0=std::chrono::system_clock::now().time_since_epoch();auto begin=std::chrono::steady_clock::now();
   y=module->forward(r,input);r.syncStream();
   const auto end=std::chrono::steady_clock::now();const auto realtime1=std::chrono::system_clock::now().time_since_epoch();
   samples.push_back({{"rep",rep},{"full_MoE_complete_wall_ms",std::chrono::duration<double,std::milli>(end-begin).count()},{"host_start_realtime_ns",std::chrono::duration_cast<std::chrono::nanoseconds>(realtime0).count()},{"host_end_realtime_ns",std::chrono::duration_cast<std::chrono::nanoseconds>(realtime1).count()}});
  }
  phase.wait();
  std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());for(auto v:bits)if((v&0x7f80)==0x7f80)throw std::runtime_error("r9264 nonfinite output");
  if(reference.empty())reference=bits;double d2=0,base2=0,maxabs=0;size_t differences=0;
  for(size_t j=0;j<bits.size();++j){auto x=fp(reference[j]),y0=fp(bits[j]);auto e=y0-x;d2+=e*e;base2+=x*x;maxabs=std::max(maxabs,std::abs(e));differences+=bits[j]!=reference[j];}
  std::vector<uint16_t>checked(storage->numel());r.memcpyD2H(checked.data(),storage->data(),storage->nbytes());if(checked!=input_original)throw std::runtime_error("r9264 readonly/tail canary");
  std::array<uint8_t,256>pre{},post{};r.memcpyD2H(pre.data(),i8storage->data(),256);r.memcpyD2H(post.data(),scratch+scratchbytes,256);for(auto v:pre)if(v!=0xa5)throw std::runtime_error("i8prefix canary");for(auto v:post)if(v!=0xa5)throw std::runtime_error("i8tail canary");if(enabled){int qerror=-1;r.memcpyD2H(&qerror,scratch+scratchbytes-4,4);if(qerror)throw std::runtime_error("i8quant nonfinite/range");}
  row["arms"].push_back({{"T",T},{"constructed_low_reuse",constructed},{"extra_scratch_bytes",scratchbytes},{"arm",arm},{"paired_comparison",arm/2},{"kind",enabled?"W2_INT8_ON":"W2_pair_baseline_OFF"},{"samples",samples},{"input_readonly_and_tail_canary",true},{"output_maxabs_vs_firstOFF",maxabs},{"output_relative_RMS_vs_firstOFF",std::sqrt(d2/std::max(base2,1e-30))},{"bit_differences_diagnostic_only",differences},{"original_outer_completion",true},{"profile_R9125_sync_OFF",true}});
 }
 if(modern_kimi_W2_pair_counts_r9280(rank,pair_after,2)||modern_kimi_I192_counts_r8669y(rank,count_after,2)||modern_kimi_serving_counts_r4665(rank,legacy_after,4))throw std::runtime_error("r9264 final counters");
 if(count_after[0]-count_before[0]!=68||count_after[1]-count_before[1]!=68*T)throw std::runtime_error("r9264 currentI192 branch not68 real calls");
 for(int j=0;j<4;++j)if(legacy_after[j]!=legacy_before[j])throw std::runtime_error("r9264 legacy384/variant4/9 executed");
 if(pair_after[0]-pair_before[0]!=0||pair_after[1]-pair_before[1]!=34)throw std::runtime_error("r9280 realOFF/ON call coverage not34each");
 if(modern_kimi_i8_counts_r113(rank,i8after,3)||i8after[0]-i8before[0]!=34||i8after[1]-i8before[1]!=34||i8after[2]-i8before[2]!=(constructed?68:0))throw std::runtime_error("i8 actual arm coverage");
 row["current_branch"].push_back({{"T",T},{"constructed_low_reuse",constructed},{"I8_OFF_calls",i8after[0]-i8before[0]},{"I8_ON_calls",i8after[1]-i8before[1]},{"I192_calls",count_after[0]-count_before[0]},{"I192_tokens",count_after[1]-count_before[1]},{"legacy_variant_calls",0},{"W2_original_OFF_calls",pair_after[0]-pair_before[0]},{"W2_pair_ON_calls",pair_after[1]-pair_before[1]}});
 }phase.wait();if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);setenv("INFINILM_KIMI_W2_PAIR_R9280","0",1);setenv("INFINILM_KIMI_W2_INT8_R113","0",1);setenv("INFINILM_KIMI_W2_CONSTRUCTED_R113","0",1);}phase.wait();module.reset();r.syncStream();row["module_closed"]=true;}r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;reports[rank]=row;
 }catch(const std::exception&e){std::fprintf(stderr,"caller8669o rank%d %s\n",rank,e.what());std::terminate();}});
 for(auto&t:threads)t.join();d["all8normal_before_communicators"]=true;}if(copyStats(copyAfter,5))throw std::runtime_error("r9280 CPUcopy read after");J copyDelta=J::array();for(size_t i=0;i<5;++i){if(copyAfter[i]<copyBefore[i])throw std::runtime_error("r9280 CPUcopy nonmonotonic");copyDelta.push_back(copyAfter[i]-copyBefore[i]);}d["CPUcopy_global_process_delta"]=copyDelta;d["CPUcopy_scope"]="entire native session including original weightsinstallation, forward and teardown; pernode aggregate not perowner or isolatedGPU-H2D timing";d["node_rank"]=node;d["expected_native_expert_shard"]={{"W13",{896,384,1792}},{"S13",{896,384,112}},{"W2",{896,3584,96}},{"S2",{896,3584,6}}};d["reports"]=reports;d["completed"]=true;result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}}
