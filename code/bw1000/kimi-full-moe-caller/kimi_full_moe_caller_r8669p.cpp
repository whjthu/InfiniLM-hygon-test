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
#include <unordered_map>
#include <mutex>
#include <condition_variable>
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);
extern "C" int modern_kimi_serving_counts_r4665(int,uint64_t*,size_t);
namespace {using J=nlohmann::json;namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
struct Barrier{std::mutex m;std::condition_variable cv;size_t count=0,generation=0;void wait(){std::unique_lock<std::mutex>lock(m);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
thread_local std::string result;
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
using Begin=void(*)(int,int);using End=const char*(*)();
}
extern "C" const char*modern_kimi_full_moe_caller_result_r8669o(){return result.c_str();}
extern "C" int modern_kimi_full_moe_caller_r8669o(const char*payload){
 J d;try{const J q=J::parse(payload);auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;
 en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;
 auto begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"kimi_caller_begin_r8669o"));auto end=reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"kimi_caller_end_r8669o"));if(!begin||!end)throw std::runtime_error("caller sampler missing");
 {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
 for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{auto info=group.get_rank_info(rank);infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"rank",rank},{"arms",J::array()}};
 {auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);auto module=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});
 std::unordered_map<std::string,rt::Tensor>weights;
 for(const auto&z:q.at("weights")){const int t=z.at("dtype");auto dtype=t==0?rt::DataType::kFloat32:t==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;weights.emplace(z.at("name"),rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),dtype,{rt::Device::Type::kCpu,0}));}
 auto keys=module->state_required_keys();if(keys.size()!=weights.size())throw std::runtime_error("fullMoE requiredweightcount");for(const auto&k:keys)if(!weights.count(k))throw std::runtime_error("fullMoE missingweight "+k);module->load_parameters_no_sync(r,weights,true);r.syncStream();post(r,*module);weights.clear();modern_kimi_set_grouped_prefill(0);modern_kimi_set_tiled_w13_r1861(1);
 auto input=rt::Tensor::empty(r,{1,16,7168},rt::DataType::kBFloat16);r.memcpyH2D(input->data(),reinterpret_cast<void*>(q.at("input_pointer").get<uintptr_t>()),input->nbytes(),false);r.syncStream();std::vector<uint16_t>gold;
 for(int arm=0;arm<5;++arm){phase.wait();bool instrumented=arm==2||arm==3;uint64_t before[4],after[4];if(modern_kimi_serving_counts_r4665(rank,before,4))throw std::runtime_error("servingcounter");
 if(instrumented)begin(rank,1);const auto start=std::chrono::steady_clock::now();auto y=module->forward(r,input);if(instrumented)begin(rank,2);r.syncStream();double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();J caller=instrumented?J::parse(end()):J();
 if(modern_kimi_serving_counts_r4665(rank,after,4))throw std::runtime_error("servingcounter");std::array<uint64_t,4>delta{};for(int i=0;i<4;++i)delta[i]=after[i]-before[i];if(delta!=std::array<uint64_t,4>{1,16,0,0})throw std::runtime_error("notcurrent639variant4provider");std::vector<uint16_t>bits(y->numel());r.memcpyD2H(bits.data(),y->data(),y->nbytes());size_t nf=0;for(auto b:bits)nf+=(b&0x7f80)==0x7f80;if(nf)throw std::runtime_error("nonfinite");if(arm==0)gold=bits;bool exact=bits==gold;
 row["arms"].push_back({{"arm",arm},{"kind",arm==0?"warm":instrumented?"instrumented":"ordinary"},{"full_MoE_complete_wall_ms",wall},{"outer_drain_included",true},{"actual_provider_delta",delta},{"ownrepeat_exact",exact},{"caller",caller}});
 // Persist actual outputs before classification; no intermediatebit Model gate.
 std::string file=q.at("raw_directory").get<std::string>()+"/rank"+std::to_string(rank)+"-arm"+std::to_string(arm)+".bin";std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);if(!f)throw std::runtime_error("outputpersist");if(!exact)row["numerical_repeat_mismatch"]=true;
 }module.reset();r.syncStream();row["module_closed"]=true;}r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;reports[rank]=row;
 }catch(const std::exception&e){std::fprintf(stderr,"caller8669o rank%d %s\n",rank,e.what());std::terminate();}});
 for(auto&t:threads)t.join();d["all8normal_before_communicators"]=true;}d["reports"]=reports;d["completed"]=true;result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}}
