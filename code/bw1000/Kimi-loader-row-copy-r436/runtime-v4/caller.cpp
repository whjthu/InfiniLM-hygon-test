#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace infinilm::runtime;
using Bytes=std::vector<std::byte>;
extern "C" int modern_cpu_copy_stats_r2515(std::uint64_t *,std::size_t);
static void require(bool x,const char *s){if(!x)throw std::runtime_error(s);}
static Tensor blob(std::shared_ptr<Bytes> b,Shape sh,Strides st,std::size_t offset=0){
 return Tensor::strided_from_blob(b->data()+offset,sh,st,DataType::kUInt8,Device{Device::Type::kCpu},b);
}
static std::shared_ptr<Bytes> bytes(std::size_t n){auto p=std::make_shared<Bytes>(n);for(std::size_t i=0;i<n;++i)(*p)[i]=std::byte((i*29+17)%251);return p;}
static void logical_equal(Tensor a,Tensor b){
 require(a->shape()==b->shape(),"shape mismatch");
 for(std::size_t i=0;i<a->shape()[0];++i)for(std::size_t k=0;k<a->shape()[1];++k)
  require(a->data()[i*a->stride(0)+k*a->stride(1)]==b->data()[i*b->stride(0)+k*b->stride(1)],"logical bytes mismatch");
}
int main(int argc,char **argv){try{
 require(argc==3,"usage caller check|bench width");
 auto rt=Runtime::create(Device{Device::Type::kCpu});
 require(rt&&rt->device().type()==Device::Type::kCpu,"not CPU runtime");
 std::uint64_t before[5]{},after[5]{};require(!modern_cpu_copy_stats_r2515(before,5),"stats before");
 int cases=0;double wall=0;const std::string operation=argv[1];const int width=std::stoi(argv[2]);
 if(operation=="check"){
  for(int w:{6,96})for(int rank=0;rank<16;++rank){
   auto p=bytes(3584*w*16+32);auto original=*p;std::weak_ptr<Bytes> weak=p;
   auto src=blob(p,{3584,std::size_t(w)},{w*16,1},rank*w);
   p.reset();require(!weak.expired(),"Tensor failed to retain external owner");
   auto dst=src->contiguous(*rt);logical_equal(src,dst);
   auto padded=bytes(3584*(w+7)+32);auto preserve=*padded;
   auto scatter=blob(padded,{3584,std::size_t(w)},{w+7,1});
   scatter->copy_from(*rt,src);logical_equal(src,scatter);
   for(std::size_t row=0;row<3584;++row)for(int col=w;col<w+7;++col)
    require((*padded)[row*(w+7)+col]==preserve[row*(w+7)+col],"scatter touched row padding");
   require(*weak.lock()==original,"source modified");
   dst.reset();scatter.reset();src.reset();require(weak.expired(),"external owner leaked");++cases;
  }
  // Actual copy_from must gather before overlapping scatter. Both views are
  // real owner-retaining Tensors; expected bytes use the pre-copy storage.
  auto p=bytes(400);auto expected=*p;
  auto src=blob(p,{8,6},{16,1});auto dst=blob(p,{8,6},{16,1},3);
  for(int i=0;i<8;++i)for(int k=0;k<6;++k)expected[3+i*16+k]=(*p)[i*16+k];
  dst->copy_from(*rt,src);require(*p==expected,"overlap gather/scatter broken");++cases;
  // Unsupported strides exercise the unmodified helper fallback.
  auto q=bytes(400);auto stepped=blob(q,{8,6},{24,2});auto out=stepped->contiguous(*rt);logical_equal(stepped,out);++cases;
  bool bad_shape=false;try{auto wrong=blob(q,{8,7},{24,2});out->copy_from(*rt,wrong);}catch(const std::runtime_error&){bad_shape=true;}
  require(bad_shape,"shape mismatch accepted");++cases;
  bool wrong_owner=false;std::thread t([&]{try{out->copy_from(*rt,stepped);}catch(const std::runtime_error&e){wrong_owner=std::string(e.what())=="Runtime execution belongs to another worker thread";}});t.join();
  require(wrong_owner,"cross-thread owner accepted");++cases;
 }else{
  require(operation=="bench"&&(width==6||width==96),"bad benchmark geometry");
  auto p=bytes(3584*width*16+32);auto src=blob(p,{3584,std::size_t(width)},{width*16,1});
  // Time the actual Runtime-owned allocation + full contiguous() caller.
  // Initialization, process launch and verification stay outside this scope.
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<128;++i){auto out=src->contiguous(*rt);require(out->data()[width-1]==src->data()[width-1],"timed guard");}
  wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  auto out=src->contiguous(*rt);logical_equal(src,out);cases=1;
 }
 require(!modern_cpu_copy_stats_r2515(after,5),"stats after");
 std::uint64_t hits[2]{};auto h=reinterpret_cast<int(*)(std::uint64_t*,std::size_t)>(dlsym(RTLD_DEFAULT,"modern_cpu_copy_row_hits_r436"));
 if(h)require(!h(hits,2),"new hits ABI");
 const bool on=std::getenv("INFINILM_CPU_ROW_COPY_R436")&&std::string(std::getenv("INFINILM_CPU_ROW_COPY_R436"))=="1";
 if(on)require(h&&hits[0]>0,"candidate failed to dispatch");
 Dl_info info{};require(dladdr(reinterpret_cast<void*>(&modern_cpu_copy_stats_r2515),&info)!=0,"provider missing");
 for(const auto &entry:std::filesystem::directory_iterator("/proc/self/fd")){
  std::error_code ec;auto name=std::filesystem::read_symlink(entry.path(),ec).string();
  require(ec||name.find("/dev/kfd")==std::string::npos,"CPU caller opened KFD");
  require(ec||name.find("/dev/dri/")==std::string::npos,"CPU caller opened DRM");
 }
 rt->syncStream();rt.reset();
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"operation\":\""<<operation<<"\",\"width\":"<<width
  <<",\"copy_calls_timed\":"<<(operation=="bench"?128:0)<<",\"caller_wall_s\":"<<wall<<",\"on\":"<<(on?"true":"false")
  <<",\"chosen\":"<<hits[0]<<",\"fallback\":"<<hits[1]<<",\"accelerator_fd_open\":false,\"provider\":\""<<info.dli_fname<<"\",\"stats_delta\":[";
 for(int i=0;i<5;++i)std::cout<<(i?",":"")<<after[i]-before[i];
 std::cout<<"]}"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
