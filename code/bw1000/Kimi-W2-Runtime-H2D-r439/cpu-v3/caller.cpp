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
#include <poll.h>
#include <fcntl.h>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <algorithm>
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

static double unix_now(){return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();}
static void write_once(const std::string&path,const Json&q){
 const std::string text=q.dump(2)+"\n";int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);require(fd>=0,"immutable output alreadyexists or cannotopen");
 std::size_t done=0;while(done<text.size()){auto n=::write(fd,text.data()+done,text.size()-done);if(n<0&&errno==EINTR)continue;if(n<=0){::close(fd);throw std::runtime_error("output write failed");}done+=std::size_t(n);}require(::close(fd)==0,"output close failed");
}
static std::string line_before(double deadline){
 std::string line;while(true){double left=deadline-unix_now();require(left>0,"absolute command deadline");pollfd p{STDIN_FILENO,POLLIN,0};int rc=::poll(&p,1,int(std::min(left*1000.0,1000.0)));if(rc<0&&errno==EINTR)continue;require(rc>=0,"stdin poll failed");if(!rc)continue;char c;auto n=::read(STDIN_FILENO,&c,1);if(n<0&&errno==EINTR)continue;require(n==1,"stdin EOF before close");if(c=='\n')return line;require(line.size()<4096,"command frame too large");line+=c;}
}
struct Command{int seq;std::string op;int width;};
static std::vector<Command> commands(const std::string&role){
 std::vector<Command> out;out.push_back({role=="baseline"?0:1,"check",0});int seq=2;
 for(int width:{96,6})for(int i=0;i<18;++i){bool candidate=i==1||(i>=2&&((i-2)%4==1||(i-2)%4==2));if((role=="candidate")==candidate)out.push_back({seq,"bench",width});++seq;}
 require(out.size()==19&&seq==38,"fixed plan construction");return out;
}
static Json work(Runtime&rt,bool on,const std::string&operation,int width,const std::string&result_path,int seq,const std::string&role,double deadline,const Json&identity,const std::string&provider){
 const auto main_start=Clock::now();Json q={{"passed",false},{"main_entered",true},{"Model",false},{"checkpoint",false},{"actual_target_gpu_count",1},{"seq",seq},{"role",role},{"PID",::getpid()},{"device",identity},{"Runtime_provider",provider},{"Runtime_address",reinterpret_cast<std::uintptr_t>(&rt)}};
 std::string phase_path=result_path+".phase.json";
 auto mark=[&](const std::string&name){
  if(phase_path.empty())return;
  Json p={{"phase",name},{"PID",::getpid()},{"elapsed_since_main_s",seconds(main_start)},{"main_entered",true}};
  if(q.contains("device"))p["device"]=q["device"];
  const std::string tmp=phase_path+".tmp";{std::ofstream out(tmp);out<<p.dump(2)<<std::endl;if(!out)throw std::runtime_error("phase write failed");}
  std::filesystem::rename(tmp,phase_path);
 };
 try{
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

  rt.syncStream();q["banks_sources_views_released"]=true;require(unix_now()<deadline,"operation completed after absolute deadline");mark("op_release_complete");q["passed"]=true;
 }catch(const std::exception&e){q["error"]=e.what();try{mark("exception");}catch(const std::exception&secondary){q["phase_save_error"]=secondary.what();}}
 q["main_wall_s"]=seconds(main_start);q["Runtime_retained"]=true;q["next_op_allowed"]=q["passed"];write_once(result_path,q);return q;
}
int main(int argc,char**argv){
 std::signal(SIGPIPE,SIG_IGN);const auto main_start=Clock::now();Json session={{"protocol",1},{"PID",::getpid()},{"passed",false},{"Model",false}};std::unique_ptr<Owned>owner;std::string directory;int last_seq=-1;std::size_t index=0;
 try{
  require(argc==4,"usage caller role output_directory absolute_deadline");const std::string role=argv[1];directory=argv[2];const double deadline=std::stod(argv[3]);
  require(role=="baseline"||role=="candidate","unknown fixed role");const bool on=role=="candidate";require(std::getenv("INFINILM_CPU_ROW_COPY_R436")&&std::string(std::getenv("INFINILM_CPU_ROW_COPY_R436"))==(on?"1":"0"),"fixed role/mode mismatch");
  require(directory.rfind("/refactor/results/Kimi439-",0)==0&&std::filesystem::is_directory(directory),"invalid immutable run output directory");require(deadline>unix_now(),"startup past deadline");
  session["role"]=role;session["on"]=on;session["deadline_unix"]=deadline;
  auto checkpoint=[&](const std::string&name){Json p={{"phase",name},{"PID",::getpid()},{"elapsed_since_main_s",seconds(main_start)}};const std::string path=directory+"/session.phase.json",tmp=path+".tmp";{std::ofstream out(tmp);out<<p.dump(2)<<std::endl;require(bool(out),"session phase write failed");}std::filesystem::rename(tmp,path);};
  checkpoint("main_entered");checkpoint("gpu_identity.before");int count=0;cuda_check(cudaGetDeviceCount(&count));require(count==1,"exactly one visible GPU required");cudaDeviceProp props{};cuda_check(cudaGetDeviceProperties(&props,0));char pci[64]{};cuda_check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));require(props.multiProcessorCount==80&&props.totalGlobalMem>=60ull*1024*1024*1024,"unexpected claimed device");
  Json identity={{"physical_PCI",pci},{"name",props.name},{"CU",props.multiProcessorCount},{"total_memory",props.totalGlobalMem},{"major",props.major},{"minor",props.minor}};session["device"]=identity;checkpoint("gpu_identity.after");
  Dl_info info{};require(dladdr(reinterpret_cast<void*>(&modern_cpu_copy_stats_r2515),&info),"Runtime provider missing");const std::string provider=info.dli_fname;session["Runtime_provider"]=provider;std::ifstream maps("/proc/self/maps");session["maps"]=std::string(std::istreambuf_iterator<char>(maps),{});
  checkpoint("Runtime_create.before");auto init=Clock::now();owner=std::make_unique<Owned>();auto&rt=*owner->rt;session["Runtime_create_s"]=seconds(init);session["Runtime_address"]=reinterpret_cast<std::uintptr_t>(&rt);checkpoint("Runtime_create.after");require(unix_now()<deadline,"ready after absolute deadline");
  session["type"]="ready";session["passed"]=true;session["main_to_ready_s"]=seconds(main_start);write_once(directory+"/ready.json",session);std::cout<<session.dump()<<std::endl;
  auto expected=commands(role);
  while(true){
   Json cmd=Json::parse(line_before(deadline));require(cmd.is_object()&&cmd.size()==6,"exact command schema");require(cmd.at("protocol")==1&&cmd.at("role")==role&&std::abs(cmd.at("deadline_unix").get<double>()-deadline)<1e-6,"command role/deadline/protocol mismatch");
   require(cmd.at("seq").is_number_integer()&&cmd.at("width").is_number_integer()&&cmd.at("op").is_string(),"command types");int seq=cmd.at("seq");std::string op=cmd.at("op");require(seq>last_seq,"replayed or reordered command");
   if(op=="close"){
    require(index==expected.size()&&seq==(on?39:38)&&cmd.at("width")==0,"close before full fixed plan");checkpoint("normal_destroy.before");auto close_start=Clock::now();rt.syncStream();std::weak_ptr<Runtime>weak=owner->rt;owner->rt.reset();require(weak.expired(),"Runtime retained at close");
    Json reply={{"protocol",1},{"type","closed"},{"seq",seq},{"role",role},{"PID",::getpid()},{"passed",true},{"Runtime_expired",true},{"work_ops",index},{"drain_destroy_s",seconds(close_start)},{"process_main_wall_s",seconds(main_start)}};checkpoint("normal_destroy.after");write_once(directory+"/close.json",reply);std::cout<<reply.dump()<<std::endl;return 0;
   }
   require(index<expected.size(),"too many work commands");const auto e=expected[index];require(seq==e.seq&&op==e.op&&cmd.at("width")==e.width,"wrong work sequence or operation");
   // Width belongs to the fixed sequence, not a caller-controlled geometry.
   const std::string path=directory+"/seq"+std::to_string(seq)+"-"+op+"-"+std::to_string(e.width)+".json";
   Json result=work(rt,on,op,e.width,path,seq,role,deadline,identity,provider);result["protocol"]=1;result["type"]="result";result["path"]=path;std::cout<<result.dump()<<std::endl;
   require(result["passed"].get<bool>(),"operation failed; no next command");last_seq=seq;++index;
  }
 }catch(const std::exception&e){
  Json error={{"protocol",1},{"type","failed"},{"PID",::getpid()},{"passed",false},{"error",e.what()},{"last_seq",last_seq},{"work_ops",index},{"process_main_wall_s",seconds(main_start)}};
  if(session.contains("role"))error["role"]=session["role"];
  if(owner&&owner->rt){try{owner->rt->syncStream();std::weak_ptr<Runtime>weak=owner->rt;owner->rt.reset();error["Runtime_expired"]=weak.expired();}catch(const std::exception&cleanup){error["cleanup_error"]=cleanup.what();}}
  try{if(!directory.empty())write_once(directory+"/terminal-error.json",error);}catch(const std::exception&secondary){error["save_error"]=secondary.what();}
  std::cout<<error.dump()<<std::endl;return 1;
 }
}
