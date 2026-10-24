#include "runtime/runtime.hpp"
#include "layers/linear/dense_prefill_f32_r2475.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
namespace rt=infinilm::runtime;
namespace {
thread_local std::string last_error;
struct Context {std::shared_ptr<rt::Runtime> r;rt::Tensor weight;};
void drain(Context*c){if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
}
extern "C" const char* kimi_qkv_error_r8545(){return last_error.c_str();}
extern "C" void* kimi_qkv_create_r8545(int device,const void*cpu_weight){
 std::unique_ptr<Context>c;
 try{
  if(device<0||device>7||!cpu_weight)throw std::invalid_argument("device/CPU weight");
  c=std::make_unique<Context>();c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();
  c->weight=rt::Tensor::empty(*c->r,{4608,7168},rt::DataType::kBFloat16);
  auto source=rt::Tensor::from_blob(const_cast<void*>(cpu_weight),{4608,7168},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  c->weight->copy_from(*c->r,source);c->r->syncStream();return c.release();
 }catch(const std::exception&e){last_error=e.what();drain(c.get());return nullptr;}
}
extern "C" int kimi_qkv_run_r8545(void*raw,size_t m,const void*cpu_input,void*cpu_output,size_t chunk,double*seconds){
 auto*c=static_cast<Context*>(raw);rt::Tensor x,y,source,wt;
 try{
  if(!c||!cpu_input||!cpu_output||!seconds||(m!=512&&m!=2048)||(chunk!=0&&chunk!=512))throw std::invalid_argument("finite actual geometry only");
  auto&r=*c->r;r.assertOwnerThread();
  source=rt::Tensor::from_blob(const_cast<void*>(cpu_input),{m,7168},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  x=rt::Tensor::empty(r,{m,7168},rt::DataType::kBFloat16);x->copy_from(r,source);
  y=rt::Tensor::empty(r,{m,4608},rt::DataType::kBFloat16);wt=c->weight->permute({1,0});r.syncStream();
  auto start=std::chrono::steady_clock::now();
  if(!chunk)infinilm::layers::linear::dense_prefill_f32_r2475(r,x,wt,y,1.0F);
  else for(size_t offset=0;offset<m;offset+=chunk){
   auto a=x->narrow({{0,offset,chunk}}),b=y->narrow({{0,offset,chunk}});
   infinilm::layers::linear::dense_prefill_f32_r2475(r,a,wt,b,1.0F);
  }
  r.syncStream();*seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  r.memcpyD2H(cpu_output,y->data(),y->nbytes());r.syncStream();return 0;
 }catch(const std::exception&e){last_error=e.what();drain(c);return 1;}
}
extern "C" int kimi_qkv_destroy_r8545(void*raw){auto*c=static_cast<Context*>(raw);if(!c)return 0;drain(c);c->weight={};drain(c);delete c;return 0;}
