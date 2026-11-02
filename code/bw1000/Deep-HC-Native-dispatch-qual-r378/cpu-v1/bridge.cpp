// HC dispatch/lifetime bridge. All HC math is the sole linked Native's pre().
#include "deepseek_v4_hyper_connection.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <stdexcept>
#include <string>
namespace rt=infinilm::runtime;
namespace m=infinilm::models::deepseek_v4;
namespace {
thread_local std::string error;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context{
 std::shared_ptr<rt::Runtime> owner;
 std::unique_ptr<m::HyperConnection> hc;
 rt::Tensor staging;
 int device;
 Context(int d,double eps,double hceps,int it,const float*fn,const float*base,const float*scale):owner(rt::Runtime::create({rt::Device::Type::kHygon,d})),device(d){
  owner->enableOwnerDrainFailStop();auto&r=*owner;
  if(!fn||!base||!scale)throw std::invalid_argument("weights");
  hc=std::make_unique<m::HyperConnection>(r,eps,hceps,it);
  auto from=[&](const float*p,rt::Shape shape){return rt::Tensor::from_blob(const_cast<float*>(p),shape,rt::DataType::kFloat32,{rt::Device::Type::kCpu,0});};
  hc->load_parameter(r,"fn",from(fn,{24,16384}));hc->load_parameter(r,"base",from(base,{24}));hc->load_parameter(r,"scale",from(scale,{3}));hc->process_weights_after_loading(r,false);
  staging=rt::Tensor::empty(r,{1,4,4096},rt::DataType::kBFloat16);r.syncStream();
 }
};
void owned(Context*c){if(!c)throw std::invalid_argument("context");c->owner->assertOwnerThread();int d=-1;ck(cudaGetDevice(&d));if(d!=c->device||c->owner->isGraphRecording())throw std::invalid_argument("owner/device/eager");}
}
extern "C" const char*deep_hc_bridge_error_r378(){return error.c_str();}
extern "C" void*deep_hc_bridge_create_r378(int d,double eps,double hceps,int it,const float*fn,const float*base,const float*scale){try{infini::rt::DeviceGuard g{rt::Device::Type::kHygon,d,true};return new Context(d,eps,hceps,it,fn,base,scale);}catch(const std::exception&e){error=e.what();return nullptr;}}
// BF16 input[1,4,4096], BF16 collapsed[1,4096], F32 post[1,4], F32 comb[1,4,4].
extern "C" int deep_hc_bridge_forward_r378(void*raw,void*const*p,void*producer){auto*c=static_cast<Context*>(raw);try{
 owned(c);if(!p)throw std::invalid_argument("buffers");auto&r=*c->owner;ck(cudaStreamSynchronize(static_cast<cudaStream_t>(producer)));
 auto wrap=[&](int i,rt::Shape shape,rt::DataType dt){if(!p[i])throw std::invalid_argument("nullbuffer");cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p[i]));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device)throw std::invalid_argument("bufferdevice");return rt::Tensor::from_blob(p[i],shape,dt,r.device());};
 auto x=wrap(0,{1,4,4096},rt::DataType::kBFloat16),collapsed=wrap(1,{1,4096},rt::DataType::kBFloat16),post=wrap(2,{1,4},rt::DataType::kFloat32),comb=wrap(3,{1,4,4},rt::DataType::kFloat32);
 c->staging->copy_from(r,x);
 // Direct external Native dispatch, no adapter-side candidate or HC math.
 auto value=c->hc->pre(r,c->staging);
 collapsed->copy_from(r,value.collapsed);post->copy_from(r,value.post);comb->copy_from(r,value.comb);r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(c&&c->owner)try{c->owner->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int deep_hc_bridge_state_r378(void*raw,void*const*p){auto*c=static_cast<Context*>(raw);try{owned(c);if(!p)throw std::invalid_argument("state");auto state=c->hc->state_dict();rt::Tensor ts[]={state.at("fn").tensor(),state.at("base").tensor(),state.at("scale").tensor()};for(int i=0;i<3;++i){if(!p[i])throw std::invalid_argument("statebuffer");c->owner->memcpyD2H(p[i],ts[i]->data(),ts[i]->nbytes());}c->owner->syncStream();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_hc_bridge_destroy_r378(void*raw,int*expired){auto*c=static_cast<Context*>(raw);try{owned(c);if(!expired)throw std::invalid_argument("expired");std::weak_ptr<rt::Runtime>w=c->owner;c->owner->syncStream();delete c;*expired=w.expired();return *expired?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
