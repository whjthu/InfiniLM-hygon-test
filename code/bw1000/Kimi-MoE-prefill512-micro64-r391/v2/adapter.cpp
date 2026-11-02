// One HOST adapter; all Model, operator, Runtime and GPU code are current8df.
#include "runtime/runtime.hpp"
#include "config/config_factory.hpp"
#include "models/kimi_k3/kimi_k3_moe.hpp"
#include "engine/distributed/communication_group.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sched.h>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" int modern_kimi_I192_counts_r8669y(int,uint64_t*,size_t);
extern "C" int modern_kimi_I192_T_counts_r9129(int,uint64_t*,size_t);
extern "C" int modern_kimi_serving_counts_r4665(int,uint64_t*,size_t);
extern "C" int modern_grouped_metadata_counts_r5858(int,uint64_t*,size_t);
extern "C" void modern_kimi_set_grouped_prefill(int);
extern "C" void modern_kimi_set_tiled_w13_r1861(int);

namespace {
using J=nlohmann::json;namespace rt=infinilm::runtime;
namespace en=infinilm::engine::distributed;namespace km=infinilm::models::kimi_k3;
using Clock=std::chrono::steady_clock;
thread_local std::string result;
struct Barrier {std::mutex mutex;std::condition_variable cv;size_t count=0,generation=0;
 void wait(){std::unique_lock<std::mutex>lock(mutex);auto g=generation;if(++count==8){count=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
void post(rt::Runtime&r,infinilm::weights::ModelWeights&m){for(auto&c:m.loading_children())post(r,*c);m.process_weights_after_loading(r,false);r.syncStream();}
template<size_t N>std::array<uint64_t,N> counts(int device,int(*f)(int,uint64_t*,size_t)){
 std::array<uint64_t,N>v{};require(f(device,v.data(),N)==0,"counter read");return v;}
template<size_t N>std::array<uint64_t,N> delta(const std::array<uint64_t,N>&a,const std::array<uint64_t,N>&b){
 std::array<uint64_t,N>d{};for(size_t i=0;i<N;++i){require(b[i]>=a[i],"counter backwards");d[i]=b[i]-a[i];}return d;}
void save(const std::string&path,const J&value){auto tmp=path+".tmp";{std::ofstream f(tmp);f<<value.dump(2)<<'\n';require(bool(f),"progress write");}require(std::rename(tmp.c_str(),path.c_str())==0,"progress rename");}
float f32(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
J compare(const std::vector<uint16_t>&got,const std::vector<uint16_t>&ref){
 require(got.size()==ref.size(),"output size");double ss=0,rr=0,maxabs=0,maxref=0;size_t nonfinite=0,changed=0;
 for(size_t i=0;i<got.size();++i){float x=f32(got[i]),y=f32(ref[i]);if(!std::isfinite(x)||!std::isfinite(y)){++nonfinite;continue;}double z=double(x)-double(y);ss+=z*z;rr+=double(y)*double(y);maxabs=std::max(maxabs,std::abs(z));maxref=std::max(maxref,std::abs(double(y)));changed+=got[i]!=ref[i];}
 double relative=std::sqrt(ss/std::max(rr,1e-30));double normalized=maxabs/std::max(maxref,1e-30);
 return{{"finite",nonfinite==0},{"nonfinite",nonfinite},{"relative_RMS",relative},{"max_abs",maxabs},{"max_abs_over_reference_max",normalized},{"changed_BF16_elements_diagnostic",changed},{"screen_pass",nonfinite==0&&relative<=1.0/64&&normalized<=1.0/8}};
}
std::vector<int> affinity(int rank){cpu_set_t available;CPU_ZERO(&available);require(sched_getaffinity(0,sizeof(available),&available)==0,"get affinity");std::vector<int>cpus;for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&available))cpus.push_back(i);require(cpus.size()>=32,"need32 allocated CPUs");cpu_set_t wanted;CPU_ZERO(&wanted);std::vector<int>used;for(int i=rank*4;i<rank*4+4;++i){CPU_SET(cpus.at(i),&wanted);used.push_back(cpus.at(i));}require(sched_setaffinity(0,sizeof(wanted),&wanted)==0,"set affinity");return used;}
void vote(rt::Runtime&r,infinicclComm_t comm,bool good){auto v=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);float x=good?1.f:0.f,y=0;r.memcpyH2D(v->data(),&x,4,false);r.syncStream();en::allreduce_(r,v,v,infinicclSum,comm);r.syncStream();r.memcpyD2H(&y,v->data(),4);require(y==16,"all16 vote failed");}
J routing(rt::Runtime&r,const infinilm::layers::moe::TopKRouter&router,const rt::Tensor&input,int width){
 std::array<uint64_t,896>all{};J chunks=J::array();std::vector<int32_t>ordered;
 for(int start=0;start<512;start+=width){auto x=input->narrow({{1,size_t(start),size_t(width)}})->view({size_t(width),7168});auto [weights,ids]=router.forward(r,x);
  require(ids->shape()==rt::Shape{size_t(width),16}&&ids->dtype()==rt::DataType::kInt32&&weights->dtype()==rt::DataType::kFloat32,"public router geometry");r.syncStream();std::vector<int32_t>v(width*16);std::vector<float>w(width*16);r.memcpyD2H(v.data(),ids->data(),v.size()*4);r.memcpyD2H(w.data(),weights->data(),w.size()*4);std::array<uint64_t,896>hist{};
  for(size_t i=0;i<v.size();++i){require(v[i]>=0&&v[i]<896&&std::isfinite(w[i]),"router id/weight invalid");++hist[v[i]];++all[v[i]];}ordered.insert(ordered.end(),v.begin(),v.end());size_t unique=0,maximum=0,jobs=0,lower=0;for(size_t e=0;e<896;++e){unique+=hist[e]>0;maximum=std::max(maximum,size_t(hist[e]));jobs+=(hist[e]+1)/2;if(e<448)lower+=hist[e];}
  chunks.push_back({{"start",start},{"rows",width},{"unique_experts",unique},{"max_expert_rows",maximum},{"TT2_jobs_proxy",jobs},{"lower448_routes",lower},{"upper448_routes",size_t(width)*16-lower}});
 }
 size_t unique=0,maximum=0;for(auto n:all){unique+=n>0;maximum=std::max(maximum,size_t(n));}
 return{{"rows",512},{"chunk_width",width},{"global_unique_experts",unique},{"global_max_expert_rows",maximum},{"histogram896",all},{"chunks",chunks},{"ordered_IDs",ordered},{"outside_timer",true},{"source","same public current TopKRouter/config/parameters, separate observer instance"}};
}
}

extern "C" const char* kimi_moe512_micro64_result_r391(){return result.c_str();}
extern "C" int kimi_moe512_micro64_r391(const char*payload){
 J final;try{
  J q=J::parse(payload);require(q.at("schema")==1&&q.at("input_pointers").size()==2&&q.at("plan").is_array(),"payload schema");
  int node=q.at("node_rank");require(node==0||node==1,"two nodes only");auto cfg=infinilm::config::ConfigFactory::createConfig(q.at("config").dump());auto normalized=cfg->get_config_json();
  en::DistConfig topology(8);topology.pp_size=1;topology.pp_stage=0;topology.tp_node_size=2;topology.tp_node_rank=node;topology.master_addr=q.at("master_addr");topology.master_port=q.at("master_port");
  std::array<J,8>reports;std::vector<std::thread>threads;Barrier phase;
  {en::CommunicationGroup group(topology,rt::Device::Type::kHygon);
   for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{J row;std::string stem=q.at("raw_directory").get<std::string>()+"/rank"+std::to_string(rank);try{
    row={{"rank",rank},{"node_rank",node},{"global_rank",node*8+rank},{"completed",false},{"trials",J::array()}};row["CPU_affinity"]=affinity(rank);auto info=group.get_rank_info(rank);
    require(info.tp_size==16&&info.tp_rank==node*8+rank&&info.comm&&info.pp_size==1&&info.pp_stage==0,"true globalTP16 RankInfo");infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();
    {auto config=std::make_shared<infinilm::config::ModelConfig>(normalized);auto module=std::make_unique<km::KimiK3MoE>(r,config,1,info,infinilm::backends::AttentionBackend::PAGED_ATTN,infinilm::layers::moe::EPConfig{});auto observer=std::make_unique<infinilm::layers::moe::TopKRouter>(r,config);
     std::unordered_map<std::string,rt::Tensor>weights;for(const auto&z:q.at("weights")){int kind=z.at("dtype");require(kind>=0&&kind<=2,"weight dtype");auto type=kind==0?rt::DataType::kFloat32:kind==1?rt::DataType::kBFloat16:rt::DataType::kUInt8;std::string name=z.at("name");require(weights.emplace(name,rt::Tensor::from_blob(reinterpret_cast<void*>(z.at("pointer").get<uintptr_t>()),z.at("shape").get<rt::Shape>(),type,{rt::Device::Type::kCpu,0})).second,"duplicate weight");}
     auto keys=module->state_required_keys();require(keys.size()==weights.size(),"fullMoE exact keycount");for(const auto&k:keys)require(weights.count(k),"missing MoE weight");module->load_parameters_no_sync(r,weights,true);post(r,*module);
     std::unordered_map<std::string,rt::Tensor>gate;for(const auto&k:observer->state_required_keys()){require(weights.count("gate."+k),"public router parameter mapping");gate.emplace(k,weights.at("gate."+k));}observer->load_parameters_no_sync(r,gate,true);post(r,*observer);weights.clear();gate.clear();
     // These selectors are TLS: set them on this exact Runtime-owning thread.
     modern_kimi_set_grouped_prefill(1);modern_kimi_set_tiled_w13_r1861(0);row["owner_TLS_grouped_prefill"]=1;row["owner_TLS_tiled_W13"]=0;
     for(int fixture=0;fixture<2;++fixture){
      auto storage=rt::Tensor::empty(r,{1,513,7168},rt::DataType::kBFloat16);std::vector<uint16_t>original(513*7168,uint16_t{0x3f80});std::memcpy(original.data(),reinterpret_cast<void*>(q.at("input_pointers")[fixture].get<uintptr_t>()),512*7168*2);r.memcpyH2D(storage->data(),original.data(),storage->nbytes(),false);r.syncStream();auto input=storage->narrow({{1,0,512}});
      row["routing"][std::to_string(fixture)]={{"T512",routing(r,*observer,input,512)},{"eightT64",routing(r,*observer,input,64)}};save(stem+"-progress.json",row);
      std::array<std::vector<uint16_t>,3>own;std::vector<uint16_t>reference;
      for(const auto&record:q.at("plan")){
       if(record.at("fixture").get<int>()!=fixture)continue;int arm=record.at("strategy");require(arm>=0&&arm<=2,"strategyABC");bool warm=record.at("warm");
       require(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())<q.at("deadline_unix").get<double>(),"original residency deadline");
       phase.wait();if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y",arm==1?"0":"1",1);}phase.wait();vote(r,info.comm,true);
       auto before=counts<2>(rank,modern_kimi_I192_counts_r8669y);auto tb=counts<65>(rank,modern_kimi_I192_T_counts_r9129);auto hb=counts<7>(rank,modern_grouped_metadata_counts_r5858);auto sb=counts<4>(rank,modern_kimi_serving_counts_r4665);
       auto outguard=rt::Tensor::empty(r,{1,513,7168},rt::DataType::kBFloat16);std::vector<uint16_t>poison(513*7168,uint16_t{0x7fc0});std::fill(poison.end()-7168,poison.end(),uint16_t{0x3f80});r.memcpyH2D(outguard->data(),poison.data(),outguard->nbytes(),false);r.syncStream();auto output=outguard->narrow({{1,0,512}});std::vector<rt::Tensor>held;
       auto tick=Clock::now();
       if(arm==0){auto y=module->forward(r,input);output->copy_from(r,y);held.push_back(y);}
       else for(size_t start=0;start<512;start+=64){auto chunk=input->narrow({{1,start,64}});auto y=module->forward(r,chunk);output->narrow({{1,start,64}})->copy_from(r,y);held.push_back(y);}
       held.clear();std::vector<rt::Tensor>().swap(held);r.syncStream();double wall=std::chrono::duration<double,std::milli>(Clock::now()-tick).count();
       auto cd=delta(before,counts<2>(rank,modern_kimi_I192_counts_r8669y));auto td=delta(tb,counts<65>(rank,modern_kimi_I192_T_counts_r9129));auto hd=delta(hb,counts<7>(rank,modern_grouped_metadata_counts_r5858));auto sd=delta(sb,counts<4>(rank,modern_kimi_serving_counts_r4665));bool counters=cd==std::array<uint64_t,2>{arm==2?8u:0u,arm==2?512u:0u}&&hd[0]==uint64_t(arm==0)&&hd[1]==uint64_t(arm==0?512:0)&&sd==std::array<uint64_t,4>{};for(size_t t=0;t<65;++t)counters&=td[t]==uint64_t(arm==2&&t==64?8:0);
       std::vector<uint16_t>got(outguard->numel()),checked(storage->numel());r.memcpyD2H(got.data(),outguard->data(),outguard->nbytes());r.memcpyD2H(checked.data(),storage->data(),storage->nbytes());bool canary=std::all_of(got.end()-7168,got.end(),[](uint16_t v){return v==0x3f80;});got.resize(512*7168);bool readonly=checked==original;if(reference.empty()){require(arm==0&&warm,"A warm reference first");reference=got;}if(own[arm].empty())own[arm]=got;J error=compare(got,reference),repeat=compare(got,own[arm]);
       J item=record;item.update({{"full_strategy_wall_ms",wall},{"MoE_calls",arm==0?1:8},{"output_stitch_and_final_drain_included",true},{"I192_counts_delta",cd},{"I192_T_hist_delta",td},{"helper_metadata_counts_delta",hd},{"legacy_serving_counts_delta",sd},{"counters_pass",counters},{"input_readonly",readonly},{"output_tail_canary",canary},{"error_vs_A",error},{"ownrepeat",repeat}});row["trials"].push_back(item);save(stem+"-progress.json",row);
       vote(r,info.comm,counters&&canary&&readonly&&error.at("screen_pass").get<bool>()&&repeat.at("screen_pass").get<bool>());
       if(warm){std::ofstream f(stem+"-f"+std::to_string(fixture)+"-strategy"+std::to_string(arm)+"-warm.bin",std::ios::binary);f.write(reinterpret_cast<const char*>(got.data()),got.size()*2);require(bool(f),"warm output persist");}
      }
     }
     phase.wait();if(rank==0)setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);phase.wait();observer.reset();module.reset();r.syncStream();row["module_and_observer_closed"]=true;
    }r.trimMemory();owner.reset();row["normal_Runtime_close"]=true;row["completed"]=true;save(stem+"-progress.json",row);reports[rank]=row;
   }catch(const std::exception&e){row["error"]=e.what();try{save(stem+"-progress.json",row);}catch(...){}std::fprintf(stderr,"r391 node%d rank%d: %s\n",node,rank,e.what());std::terminate();}});
   for(auto&t:threads)t.join();final["all8normal_before_communicators"]=true;
  }final["node_rank"]=node;final["reports"]=reports;final["completed"]=true;result=final.dump();return 0;
 }catch(const std::exception&e){final["error"]=e.what();final["completed"]=false;result=final.dump();return 1;}
}
