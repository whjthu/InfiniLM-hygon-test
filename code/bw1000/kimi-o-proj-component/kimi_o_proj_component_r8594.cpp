#include "runtime/runtime.hpp"
#include "layers/linear/linear.hpp"
#include "engine/distributed/communication_group.hpp"
#include <cuda_runtime_api.h>
#include <memory>
#include <cstring>
#include <string>
#include <chrono>
#include <stdexcept>
#include <vector>
namespace rt=infinilm::runtime;
namespace dist=infinilm::engine::distributed;
namespace {
thread_local std::string error;
// Only exposes the existing protected call; no replacement GEMM/quantization.
struct ExposedRow final:infinilm::nn::RowParallelLinear {
 using infinilm::nn::RowParallelLinear::RowParallelLinear;
 rt::Tensor local(rt::Runtime&r,rt::Tensor&x)const{return compute_linear(r,x);}
};
struct Context{std::shared_ptr<rt::Runtime>r;infinicclComm_t comm=nullptr;std::unique_ptr<ExposedRow>row;};
void drain(Context*c){if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
}
extern "C" const char*kimi_oproj_error_r8582(){return error.c_str();}
extern "C" size_t kimi_oproj_id_bytes_r8582(){return sizeof(infinicclUniqueId);}
extern "C" int kimi_oproj_id_r8582(void*out){try{if(!out)throw std::invalid_argument("null ID");infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)throw std::runtime_error("unique ID");std::memcpy(out,&id,sizeof(id));return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" void*kimi_oproj_create_r8582(int device,int rank,const void*id,const void*weight){
 std::unique_ptr<Context>c;
 try{
  if(device!=rank||rank<0||rank>7||!id||!weight)throw std::invalid_argument("actual TP8 rank/device/weight");
  if(cudaSetDevice(device)!=cudaSuccess)throw std::runtime_error("select owner device");
  c=std::make_unique<Context>();c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();
  infinicclUniqueId value{};std::memcpy(&value,id,sizeof(value));if(infinicclCommInitRank(&c->comm,8,value,rank)!=infinicclSuccess)throw std::runtime_error("TP8 CCL init");
  c->row=std::make_unique<ExposedRow>(*c->r,12288,7168,false,rt::DataType::kBFloat16,rank,8,c->comm);
  auto w=c->row->weight();if(w->shape()!=rt::Shape{7168,1536}||w->dtype()!=rt::DataType::kBFloat16)throw std::runtime_error("actual local weight shape");
  auto source=rt::Tensor::from_blob(const_cast<void*>(weight),{7168,1536},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});w->copy_from(*c->r,source);c->r->syncStream();return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());if(c){c->row.reset();drain(c.get());if(c->comm){infinicclCommDestroy(c->comm);c->comm=nullptr;}c->r.reset();}return nullptr;}
}
extern "C" int kimi_oproj_run_r8582(void*raw,size_t m,const void*input,void*local,void*output,int mode,double*seconds){
 auto*c=static_cast<Context*>(raw);rt::Tensor x,y,source;std::vector<rt::Tensor>held;
 try{
  if(!c||!input||!output||!seconds||(m!=512&&m!=2048)||mode<0||mode>3||(mode&& !local))throw std::invalid_argument("bounded actual O512/N2048 component");
  auto&r=*c->r;r.assertOwnerThread();source=rt::Tensor::from_blob(const_cast<void*>(input),{1,m,1536},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});
  x=rt::Tensor::empty(r,{1,m,1536},rt::DataType::kBFloat16);x->copy_from(r,source);r.syncStream();const auto start=std::chrono::steady_clock::now();
  if(mode==0)y=c->row->forward(r,x); // Unmodified actual RowParallel baseline.
  else{
   if(mode==2){
    y=rt::Tensor::empty(r,{1,m,7168},rt::DataType::kBFloat16);
    for(size_t pos=0;pos<m;pos+=512){auto part=x->narrow({{1,pos,512}});auto value=c->row->local(r,part);held.push_back(value);y->narrow({{1,pos,512}})->copy_from(r,value);}
   }else y=c->row->local(r,x);
   // The one new diagnostic boundary: actual local result BEFORE original CCL.
   r.syncStream();r.memcpyD2H(local,y->data(),y->nbytes());r.syncStream();
   if(mode==3){
    if(!y->is_contiguous())throw std::logic_error("expected original contiguous local output");
    auto matrix=y->view({m,7168});
    if(matrix->data()!=y->data()||!matrix->is_contiguous()||matrix->strides().size()!=2||matrix->strides()[0]!=7168||matrix->strides()[1]!=1)throw std::logic_error("zero-copy contiguous matrix view required");
    held.push_back(matrix);
    for(size_t pos=0;pos<m;pos+=512){
     auto part=matrix->narrow({{0,pos,512}});held.push_back(part);
     if(!part->is_contiguous()||part->data()!=y->data()+pos*7168*2)throw std::logic_error("actual contiguous zero-copy512 row slice required");
     dist::allreduce_(r,part,part,infinicclSum,c->comm);
    }
   }
   else dist::allreduce_(r,y,y,infinicclSum,c->comm);
  }
  r.syncStream();*seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();r.memcpyD2H(output,y->data(),y->nbytes());r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}
}
extern "C" int kimi_oproj_destroy_r8582(void*raw){auto*c=static_cast<Context*>(raw);if(!c)return 0;try{c->r->assertOwnerThread();drain(c);c->row.reset();drain(c);if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("TP8 CCL destroy");c->comm=nullptr;c->r.reset();delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
