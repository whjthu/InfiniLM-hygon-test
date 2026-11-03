#include "contract.hpp"
#include "runtime/runtime.hpp"
#include "layers/linear/linear.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <string>
#include <vector>
namespace rt=infinilm::runtime;
extern "C" int kimi_lowrank_gemv_r432(int,void*,const void*,const void*,size_t,size_t,size_t,void*);
namespace {
thread_local std::string error;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
struct Context{std::shared_ptr<rt::Runtime>r;std::unique_ptr<infinilm::nn::Linear>layer[2];cudaEvent_t begin=nullptr,end=nullptr;std::thread::id owner;int device=0;bool poison=false;uint64_t counts[4]{};};
void drain(Context*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}
void owner(Context*c){if(!c||c->poison||c->owner!=std::this_thread::get_id())throw std::invalid_argument("432 null/poison/wrong owner");int dev=-1;if(cudaGetDevice(&dev)!=cudaSuccess||dev!=c->device)throw std::invalid_argument("432 device changed");c->r->assertOwnerThread();}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Work{Context*c;rt::Tensor x,y,backing,host_x,host_y;std::vector<uint16_t>guard;~Work(){drain(c);}};
void release(Context*c)noexcept{if(!c)return;drain(c);c->layer[0].reset();c->layer[1].reset();drain(c);if(c->begin)cudaEventDestroy(c->begin);if(c->end)cudaEventDestroy(c->end);c->r.reset();}
}
extern "C" const char*kimi_lowrank_error_r432(){return error.c_str();}
extern "C" void*kimi_lowrank_create_r432(int device,const void*down,const void*up){
 std::unique_ptr<Context>c;
 try{if(device!=0||!down||!up)throw std::invalid_argument("432 single visible device0 + two complete BF16 checkpoint weights");check(cudaSetDevice(device));c=std::make_unique<Context>();c->device=device;c->owner=std::this_thread::get_id();c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();
  for(int p=0;p<2;++p){auto g=kimi_r432::geometry(p);c->layer[p]=std::make_unique<infinilm::nn::Linear>(*c->r,g.k,g.n,false,rt::DataType::kBFloat16);auto w=c->layer[p]->weight();if(w->shape()!=rt::Shape{g.n,g.k}||w->dtype()!=rt::DataType::kBFloat16||w->strides()!=rt::Strides{static_cast<rt::Stride>(g.k),1})throw std::runtime_error("432 actual replicated weight layout");auto source=rt::Tensor::from_blob(const_cast<void*>(p?up:down),{g.n,g.k},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});c->layer[p]->load_parameter(*c->r,"weight",source);c->r->syncStream();c->layer[p]->process_weights_after_loading(*c->r,false);c->r->syncStream();auto loaded=c->layer[p]->weight();if(loaded->shape()!=rt::Shape{g.n,g.k}||loaded->strides()!=rt::Strides{static_cast<rt::Stride>(g.k),1}||loaded->dtype()!=rt::DataType::kBFloat16)throw std::runtime_error("432 loaded original weight dtype/stride changed");}
  check(cudaEventCreate(&c->begin));check(cudaEventCreate(&c->end));return c.release();
 }catch(const std::exception&e){error=e.what();release(c.get());return nullptr;}
}
// metrics: inputseed_s, allocation+dispatch+drain_s, event_ms, outputreadback_s,
// complete_call_s, canary_ok. event interval may include HOST submission gaps.
extern "C" int kimi_lowrank_run_r432(void*raw,int projection,int mode,const void*input,void*output,int guarded,double*metrics){
 auto*c=static_cast<Context*>(raw);auto total=Clock::now();
 if(!c||c->owner!=std::this_thread::get_id()){error="432 null/wrong owner rejected before Runtime";return 1;}
 int selected=-1;if(cudaGetDevice(&selected)!=cudaSuccess||selected!=c->device){error="432 wrong device rejected before Runtime";return 1;}
 try{owner(c);auto g=kimi_r432::geometry(projection);kimi_r432::validate(projection,mode,g.n,g.k,g.k);if(!input||!output||!metrics||(guarded!=0&&guarded!=1))throw std::invalid_argument("432 buffer/guard contract");
  {Work w{c};auto&r=*c->r;w.host_x=rt::Tensor::from_blob(const_cast<void*>(input),{1,g.k},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});auto seed=Clock::now();w.x=rt::Tensor::empty(r,{1,g.k},rt::DataType::kBFloat16);w.x->copy_from(r,w.host_x);
   if(guarded){w.guard.assign(g.n+32,0x3f25);w.backing=rt::Tensor::empty(r,{1,g.n+32},rt::DataType::kBFloat16);auto host=rt::Tensor::from_blob(w.guard.data(),{1,g.n+32},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});w.backing->copy_from(r,host);}
   r.syncStream();metrics[0]=seconds(seed);auto dispatch=Clock::now();check(cudaEventRecord(c->begin,static_cast<cudaStream_t>(r.stream())));
   if(mode==0){if(guarded)throw std::invalid_argument("432 guarded direct-kernel diagnostic is ON only");w.y=c->layer[projection]->forward(r,w.x);}
   else{w.y=guarded?w.backing->narrow({{1,16,g.n}}):rt::Tensor::empty(r,{1,g.n},rt::DataType::kBFloat16);auto weight=c->layer[projection]->weight();int rc=kimi_lowrank_gemv_r432(projection,w.y->data(),w.x->data(),weight->data(),g.n,g.k,g.k,r.stream());if(rc)throw std::runtime_error("432 GEMV launch rc="+std::to_string(rc));}
   check(cudaEventRecord(c->end,static_cast<cudaStream_t>(r.stream())));r.syncStream();metrics[1]=seconds(dispatch);float ms=0;check(cudaEventElapsedTime(&ms,c->begin,c->end));metrics[2]=ms;
   auto readback=Clock::now();w.host_y=rt::Tensor::from_blob(output,{1,g.n},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});w.host_y->copy_from(r,w.y);r.syncStream();metrics[5]=1;
   if(guarded){auto host=rt::Tensor::from_blob(w.guard.data(),{1,g.n+32},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});host->copy_from(r,w.backing);r.syncStream();for(size_t i=0;i<16;++i)if(w.guard[i]!=0x3f25||w.guard[g.n+16+i]!=0x3f25)metrics[5]=0;}
   metrics[3]=seconds(readback);c->counts[projection*2+mode]++;}
  drain(c);metrics[4]=seconds(total);return 0;
 }catch(const std::exception&e){error=e.what();if(c)c->poison=true;drain(c);return 1;}
}
extern "C" int kimi_lowrank_counts_r432(void*raw,uint64_t*out,size_t n){auto*c=static_cast<Context*>(raw);try{owner(c);if(!out||n!=4)throw std::invalid_argument("432 counts");std::memcpy(out,c->counts,sizeof(c->counts));return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int kimi_lowrank_destroy_r432(void*raw){auto*c=static_cast<Context*>(raw);if(!c){error="432 null destroy";return 1;}if(c->owner!=std::this_thread::get_id()){error="432 wrong destroy owner";return 1;}release(c);delete c;return 0;}

extern "C" int kimi_lowrank_layout_r432(void*raw,uint64_t*out,size_t n){auto*c=static_cast<Context*>(raw);try{owner(c);if(!out||n!=10)throw std::invalid_argument("432 layout buffer");for(int p=0;p<2;++p){auto w=c->layer[p]->weight();out[p*5]=w->size(0);out[p*5+1]=w->size(1);out[p*5+2]=w->strides()[0];out[p*5+3]=w->strides()[1];out[p*5+4]=(w->dtype()==rt::DataType::kBFloat16);}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
