#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <unistd.h>
using namespace infinilm::runtime;
using Json=nlohmann::json;
using Bytes=std::vector<std::byte>;
using Clock=std::chrono::steady_clock;
extern "C" int modern_cpu_copy_stats_r2515(std::uint64_t*,std::size_t);
static void require(bool x,const char*s){if(!x)throw std::runtime_error(s);}
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void cuda_check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
struct Owned {
 std::shared_ptr<Runtime> rt;
 explicit Owned():rt(Runtime::create(Device{Device::Type::kHygon,0})){}
 ~Owned(){if(rt){try{rt->syncStream();}catch(...){std::terminate();}}}
};
static std::shared_ptr<Bytes> bytes(std::size_t n){auto p=std::make_shared<Bytes>(n);for(std::size_t i=0;i<n;++i)(*p)[i]=std::byte((i*29+17)%251);return p;}
static void unchanged(std::weak_ptr<Bytes>weak){auto p=weak.lock();require(bool(p),"source owner lost before final sync");for(std::size_t i=0;i<p->size();++i)require((*p)[i]==std::byte((i*29+17)%251),"source bytes modified");}
static Tensor cpu_blob(std::shared_ptr<Bytes> p,Shape shape,Strides strides){return Tensor::strided_from_blob(p->data(),shape,strides,DataType::kUInt8,Device{Device::Type::kCpu},p);}
static void read_check(Runtime&rt,const Tensor&gpu,const Tensor&selected){
 std::vector<std::byte> actual(gpu->nbytes());rt.memcpyD2H(actual.data(),gpu->data(),actual.size());rt.syncStream();
 for(std::size_t i=0;i<selected->size(0);++i)for(std::size_t k=0;k<selected->size(1);++k)
  require(actual[i*selected->size(1)+k]==selected->data()[i*selected->stride(0)+k*selected->stride(1)],"H2D byte mismatch");
}
static std::vector<std::uint64_t> stats(){std::vector<std::uint64_t> v(5);require(!modern_cpu_copy_stats_r2515(v.data(),5),"stats API");return v;}
static std::vector<std::uint64_t> subtract(std::vector<std::uint64_t>a,const std::vector<std::uint64_t>&b){for(std::size_t i=0;i<a.size();++i){require(a[i]>=b[i],"counter backwards");a[i]-=b[i];}return a;}
static std::vector<std::uint64_t> hits(){std::vector<std::uint64_t>v(2);auto f=reinterpret_cast<int(*)(std::uint64_t*,std::size_t)>(dlsym(RTLD_DEFAULT,"modern_cpu_copy_row_hits_r436"));if(f)require(!f(v.data(),2),"row hits API");return v;}
int main(int argc,char**argv){
 const auto main_start=Clock::now();Json q={{"passed",false},{"main_entered",true},{"Model",false},{"checkpoint",false},{"actual_target_gpu_count",1}};
 std::string phase_path=argc==4?std::string(argv[3])+".phase.json":"";
 auto mark=[&](const std::string&name){
  if(phase_path.empty())return;
  Json p={{"phase",name},{"PID",::getpid()},{"elapsed_since_main_s",seconds(main_start)},{"main_entered",true}};
  if(q.contains("device"))p["device"]=q["device"];
  const std::string tmp=phase_path+".tmp";{std::ofstream out(tmp);out<<p.dump(2)<<std::endl;if(!out)throw std::runtime_error("phase write failed");}
  std::filesystem::rename(tmp,phase_path);
 };
 try{
  mark("main_entered");
  require(argc==4,"usage caller check|bench width result.json");const std::string operation=argv[1];const int width=std::stoi(argv[2]);
  const bool on=std::getenv("INFINILM_CPU_ROW_COPY_R436")&&std::string(std::getenv("INFINILM_CPU_ROW_COPY_R436"))=="1";
  mark("gpu_identity.before");
  int count=0;cuda_check(cudaGetDeviceCount(&count));require(count==1,"exactly one visible GPU required");
  cudaDeviceProp props{};cuda_check(cudaGetDeviceProperties(&props,0));char pci[64]{};cuda_check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
  q["device"]={{"physical_PCI",pci},{"name",props.name},{"CU",props.multiProcessorCount},{"total_memory",props.totalGlobalMem},{"major",props.major},{"minor",props.minor}};
  require(props.multiProcessorCount==80&&props.totalGlobalMem>=60ull*1024*1024*1024,"unexpected claimed device");
  mark("gpu_identity.after");
  Dl_info info{};require(dladdr(reinterpret_cast<void*>(&modern_cpu_copy_stats_r2515),&info),"Runtime provider dladdr");q["Runtime_provider"]=info.dli_fname;
  std::ifstream maps("/proc/self/maps");q["maps"]=std::string(std::istreambuf_iterator<char>(maps),{});
  mark("Runtime_create.before");
  const auto init_start=Clock::now();Owned owner;auto&rt=*owner.rt;require(rt.device().type()==Device::Type::kHygon,"not Hygon Runtime");q["Runtime_create_s"]=seconds(init_start);mark("Runtime_create.after");
  auto initial_stats=stats(),initial_hits=hits();int cases=0;
  if(operation=="check"){
   require(width==0,"check width0");q["offsets"]=Json::array();
   for(int w:{96,6}){
    mark("check.width"+std::to_string(w)+".before");
    auto storage=bytes(3584*w*16);std::weak_ptr<Bytes>weak=storage;
    auto full=cpu_blob(storage,{3584,std::size_t(w*16)},{w*16,1});storage.reset();require(!weak.expired(),"source owner not retained");
    auto alloc=Clock::now();auto bank=Tensor::empty(rt,{16,3584,std::size_t(w)},DataType::kUInt8);rt.syncStream();double allocation=seconds(alloc);
    for(int rank=0;rank<16;++rank){
     auto selected=full->narrow({{1,std::size_t(rank*w),std::size_t(w)}});
     auto destination=bank->narrow({{0,std::size_t(rank),1}})->squeeze(0);
     destination->copy_from(rt,selected);rt.syncStream();read_check(rt,destination,selected);++cases;
     q["offsets"].push_back({{"width",w},{"rank",rank},{"offset",rank*w},{"bytes",3584*w},{"passed",true},{"bank_allocation_s",rank==0?allocation:0.0}});
    }
    unchanged(weak);full.reset();require(weak.expired(),"CPU source owner retained after completed transfers");bank.reset();rt.syncStream();mark("check.width"+std::to_string(w)+".after");
   }
   mark("check.fallback_shape_dtype_owner.before");
   // Original generic gather must remain available for non-unit inner stride.
   auto storage=bytes(8*24);auto stepped=cpu_blob(storage,{8,6},{24,2});auto dst=Tensor::empty(rt,{8,6},DataType::kUInt8);
   dst->copy_from(rt,stepped);rt.syncStream();read_check(rt,dst,stepped);++cases;
   bool shape=false;try{auto wrong=cpu_blob(storage,{8,7},{24,2});dst->copy_from(rt,wrong);}catch(const std::runtime_error&){shape=true;}require(shape,"shape mismatch accepted");++cases;
   bool dtype=false;try{auto wrong=Tensor::strided_from_blob(storage->data(),{8,6},{12,1},DataType::kFloat16,Device{Device::Type::kCpu},storage);dst->copy_from(rt,wrong);}catch(const std::runtime_error&){dtype=true;}require(dtype,"dtype mismatch accepted");++cases;
   bool wrong_owner=false;std::thread thread([&]{try{dst->copy_from(rt,stepped);}catch(const std::runtime_error&e){wrong_owner=std::string(e.what())=="Runtime execution belongs to another worker thread";}});thread.join();require(wrong_owner,"cross-thread owner accepted");++cases;
   rt.syncStream();dst.reset();stepped.reset();storage.reset();mark("check.fallback_shape_dtype_owner.after");
  }else{
   require(operation=="bench"&&(width==96||width==6),"bad benchmark geometry");
   mark("bench.host_setup.before");
   auto prep=Clock::now();auto storage=bytes(3584*width*16);std::weak_ptr<Bytes>weak=storage;
   auto full=cpu_blob(storage,{3584,std::size_t(width*16)},{width*16,1});storage.reset();q["host_setup_s"]=seconds(prep);
   mark("bench.bank_allocation.before");
   auto allocate=Clock::now();auto bank=Tensor::empty(rt,{896,3584,std::size_t(width)},DataType::kUInt8);rt.syncStream();q["GPU_bank_allocation_s"]=seconds(allocate);
   mark("bench.register_views.before");
   auto registration=Clock::now();std::vector<Tensor>targets;for(std::size_t e=0;e<896;++e)targets.push_back(bank->narrow({{0,e,1}})->squeeze(0));q["bank_register_views_s"]=seconds(registration);
   // Corresponds to one physical TP rank loading 896 experts. Synthetic source
   // bytes are deliberately reused; no checkpoint or cold-page claim.
   auto before=stats(),before_hits=hits();mark("bench.complete_copy.before");const auto start=Clock::now();
   for(std::size_t e=0;e<896;++e){auto selected=full->narrow({{1,0,std::size_t(width)}});targets[e]->copy_from(rt,selected);rt.syncStream();}
   q["full_gather_H2D_sync_wall_s"]=seconds(start);q["timed_stats_delta"]=subtract(stats(),before);q["timed_hits_delta"]=subtract(hits(),before_hits);mark("bench.complete_copy.after");
   q["expert_copy_calls"]=896;q["actual_H2D_bytes"]=896ull*3584*width;q["bank_logical_bytes"]=bank->nbytes();q["allocation_in_timed_wall"]=false;
   q["timed_scope"]="896 CPU narrow+temporary allocation+gather+actual copy_from H2D+internal completion sync+Weight::load-style final sync; preallocated bank/views. No D2H in timer.";
   mark("bench.D2H_validation.before");
   auto validate=Clock::now();auto selected=full->narrow({{1,0,std::size_t(width)}});for(auto&target:targets)read_check(rt,target,selected);q["all896_D2H_byte_validation_s"]=seconds(validate);cases=896;mark("bench.D2H_validation.after");
   unchanged(weak);selected.reset();targets.clear();bank.reset();full.reset();require(weak.expired(),"CPU source owner leak");rt.syncStream();
  }
  q["stats_delta"]=subtract(stats(),initial_stats);q["hits_delta"]=subtract(hits(),initial_hits);q["cases"]=cases;q["operation"]=operation;q["width"]=width;q["on"]=on;
  if(on){require(q["hits_delta"][0].get<std::uint64_t>()>0,"candidate never chosen");if(operation=="check")require(q["hits_delta"][1].get<std::uint64_t>()>0,"fallback not exercised");}
  mark("normal_destroy.before");
  auto finish=Clock::now();rt.syncStream();std::weak_ptr<Runtime>weakrt=owner.rt;owner.rt.reset();require(weakrt.expired(),"Runtime still retained after all tensors destroyed");q["drain_destroy_s"]=seconds(finish);q["Runtime_expired"]=true;mark("normal_destroy.after");q["passed"]=true;
 }catch(const std::exception&e){q["error"]=e.what();try{mark("exception");}catch(const std::exception&secondary){q["phase_save_error"]=secondary.what();}}
 q["main_wall_s"]=seconds(main_start);if(argc==4){std::ofstream out(argv[3]);out<<q.dump(2)<<std::endl;if(!out)return 2;}
 std::cout<<q.dump()<<std::endl;return q["passed"].get<bool>()?0:1;
}
