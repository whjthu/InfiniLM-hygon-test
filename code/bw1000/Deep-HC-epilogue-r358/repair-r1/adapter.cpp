// Private fixed-T1 HC-pre component. Original linked HC baseline; no Model ABI changes.
#include "deepseek_v4_hyper_connection.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <stdexcept>
extern "C" int deep_hc_epilogue_r358(const float*,const float*,const float*,const float*,const float*,void*,float*,float*,unsigned*,void*);
extern "C" int modern_deepseek_sinkhorn_r89(const float*,float*,size_t,float,int,void*,int);
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;namespace m=infinilm::models::deepseek_v4;
namespace {
thread_local std::string error,report;
using Clock=std::chrono::steady_clock;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Work{
 rt::Runtime&r;std::vector<rt::Tensor> held;
 explicit Work(rt::Runtime&owner):r(owner){}
 ~Work(){try{r.syncStream();}catch(...){std::terminate();}}
 rt::Tensor alloc(rt::Shape shape,rt::DataType dt=rt::DataType::kFloat32){auto t=rt::Tensor::empty(r,shape,dt);held.push_back(t);return t;}
 template<class Op>rt::Tensor binary(const rt::Tensor&a,const rt::Tensor&b,rt::Shape shape){auto out=alloc(shape);ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,b,out);return out;}
 template<class Op>rt::Tensor unary(const rt::Tensor&a){auto out=alloc(a->shape());ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,out);return out;}
 rt::Tensor fp32(const rt::Tensor&a){if(a->dtype()==rt::DataType::kFloat32)return a;auto out=alloc(a->shape());ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),a,out);return out;}
 rt::Tensor cast(const rt::Tensor&a,rt::DataType dt){if(a->dtype()==dt)return a;auto out=alloc(a->shape(),dt);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),a,out);return out;}
 rt::Tensor sum(const rt::Tensor&a,int64_t dim,bool keep,rt::Shape shape){auto out=alloc(shape);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),a,std::optional<std::vector<int64_t>>{{dim}},keep,std::optional<infini::ops::DataType>{},out);return out;}
};

struct Context {
 std::shared_ptr<rt::Runtime> owner;
 std::unique_ptr<m::HyperConnection> original;
 rt::Tensor fn_,base_,scale_,constants_,ones_,staging,hit;
 rt::Device device_;double rms_eps_,hc_eps_;int iterations_;
 Context(int d,double rms,double hc,int it,const float*fn,const float*base,const float*scale):owner(rt::Runtime::create({rt::Device::Type::kHygon,d})),device_(owner->device()),rms_eps_(rms),hc_eps_(hc),iterations_(it){
  owner->enableOwnerDrainFailStop();auto&r=*owner;
  if(!fn||!base||!scale)throw std::invalid_argument("null HC weights");
  original=std::make_unique<m::HyperConnection>(r,rms,hc,it);
  auto from=[&](const float*p,rt::Shape shape){return rt::Tensor::from_blob(const_cast<float*>(p),shape,rt::DataType::kFloat32,{rt::Device::Type::kCpu,0});};
  original->load_parameter(r,"fn",from(fn,{24,16384}));original->load_parameter(r,"base",from(base,{24}));original->load_parameter(r,"scale",from(scale,{3}));original->process_weights_after_loading(r,false);
  auto copy=[&](const float*p,rt::Shape shape){auto out=rt::Tensor::empty(r,shape,rt::DataType::kFloat32);r.memcpyH2D(out->data(),p,out->nbytes(),false);return out;};
  fn_=copy(fn,{24,16384});base_=copy(base,{24});scale_=copy(scale,{3});float constants[]={float(rms),float(hc),2.f};constants_=copy(constants,{3});std::vector<float>ones(16384,1.f);ones_=copy(ones.data(),{16384});staging=rt::Tensor::empty(r,{1,4,4096},rt::DataType::kBFloat16);hit=rt::Tensor::empty(r,{1},rt::DataType::kInt32);r.setDeviceMemory(hit->data(),0,4);r.syncStream();
 }
 m::HyperConnection::PreResult candidate(rt::Runtime&r,const rt::Tensor&residual){
 Work work(r);size_t t=residual->size(0);auto xf=work.fp32(residual),flat=xf->view({t,16384});
 if(t!=1)throw std::invalid_argument("358 only qualifiedT1");
 auto square=work.binary<infini::ops::Mul>(flat,flat,{t,16384}),mean=work.alloc({t,1});
 ex::submit<infini::ops::Mean>(r,infini::ops::Mean::DefaultConfig(device_.type()),square,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},mean);
 auto plus_eps=work.binary<infini::ops::Add>(mean,constants_->narrow({{0,0,1}}),{t,1});auto inv=work.unary<infini::ops::Rsqrt>(plus_eps);auto normalized=work.binary<infini::ops::Mul>(flat,inv,{t,16384});
 auto mixes=work.alloc({t,24});
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(device_.type()),normalized,fn_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},mixes);

 auto collapsed=work.alloc({1,4096},rt::DataType::kBFloat16),post=work.alloc({1,4}),comb=work.alloc({1,4,4});
 // Candidate internal poisoning is conservative measurement overhead, never elided.
 r.setDeviceMemory(collapsed->data(),0xff,collapsed->nbytes());r.setDeviceMemory(post->data(),0xff,post->nbytes());r.setDeviceMemory(comb->data(),0xff,comb->nbytes());
 int epilogue=deep_hc_epilogue_r358(reinterpret_cast<const float*>(xf->data()),reinterpret_cast<const float*>(mixes->data()),reinterpret_cast<const float*>(base_->data()),reinterpret_cast<const float*>(scale_->data()),reinterpret_cast<const float*>(constants_->data()),collapsed->data(),reinterpret_cast<float*>(post->data()),reinterpret_cast<float*>(comb->data()),reinterpret_cast<unsigned*>(hit->data()),r.stream());
 if(epilogue)throw std::runtime_error("358 HIP epilogue launch error="+std::to_string(epilogue));
 auto normalized_comb=work.alloc({t,4,4});
 int sinkhorn_status=modern_deepseek_sinkhorn_r89(reinterpret_cast<const float*>(comb->data()),reinterpret_cast<float*>(normalized_comb->data()),t,float(hc_eps_),iterations_,r.stream(),device_.index());
 if(sinkhorn_status)throw std::runtime_error(std::string("mHC fused Sinkhorn: ")+cudaGetErrorString(static_cast<cudaError_t>(sinkhorn_status)));
 comb=normalized_comb;
 r.syncStream();return {collapsed,post,comb};
 }
};
void owned(Context*c){if(!c)throw std::invalid_argument("context");c->owner->assertOwnerThread();int d=-1;ck(cudaGetDevice(&d));if(d!=c->device_.index()||c->owner->isGraphRecording())throw std::invalid_argument("owner/device/eager");}
}
extern "C" const char*deep_hc_epilogue_error_r358(){return error.c_str();}
extern "C" const char*deep_hc_epilogue_report_r358(){return report.c_str();}
extern "C" void*deep_hc_epilogue_create_r358(int d,double eps,double hceps,int it,const float*fn,const float*base,const float*scale){try{infini::rt::DeviceGuard g{rt::Device::Type::kHygon,d,true};return new Context(d,eps,hceps,it,fn,base,scale);}catch(const std::exception&e){error=e.what();return nullptr;}}
// Ordered device buffers: inputBF16[1,4,4096],collapsedBF16[1,4096],postF32[1,4],combF32[1,4,4].
// mode0 original;1 defaultOFF controlled candidate, allshapesfixed, parameters immutable aftercreate.
extern "C" int deep_hc_epilogue_forward_r358(void*raw,int mode,void*const*p,void*producer){auto*c=static_cast<Context*>(raw);try{owned(c);if(!p||mode<0||mode>1)throw std::invalid_argument("mode");auto&r=*c->owner;ck(cudaStreamSynchronize(static_cast<cudaStream_t>(producer)));
 auto wrap=[&](int i,rt::Shape shape,rt::DataType dt){cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p[i]));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device_.index())throw std::invalid_argument("bufferdevice");return rt::Tensor::from_blob(p[i],shape,dt,r.device());};
 auto x=wrap(0,{1,4,4096},rt::DataType::kBFloat16),collapsed=wrap(1,{1,4096},rt::DataType::kBFloat16),post=wrap(2,{1,4},rt::DataType::kFloat32),comb=wrap(3,{1,4,4},rt::DataType::kFloat32);
 r.setDeviceMemory(c->hit->data(),0,4);r.syncStream();auto begin=Clock::now();c->staging->copy_from(r,x);const char*flag=std::getenv("INFINILM_DEEP_HC_EPILOGUE_R358");if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))throw std::invalid_argument("HC epilogue flag");bool selected=mode==1&&flag&&std::strcmp(flag,"1")==0;
 auto value=selected?c->candidate(r,c->staging):c->original->pre(r,c->staging);collapsed->copy_from(r,value.collapsed);post->copy_from(r,value.post);comb->copy_from(r,value.comb);r.syncStream();auto end=Clock::now();report=nlohmann::json{{"candidate",selected},{"T",1},{"original_norm_and_F32Gemm",true},{"original_Sinkhorn",true},{"candidate_poison_inside_wall",selected},{"HCpre_staging_outputcopy_drained_seconds",std::chrono::duration<double>(end-begin).count()}}.dump();return 0;
 }catch(const std::exception&e){error=e.what();if(c&&c->owner)try{c->owner->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int deep_hc_epilogue_destroy_r358(void*raw,int*expired){auto*c=static_cast<Context*>(raw);try{owned(c);if(!expired)throw std::invalid_argument("expired");std::weak_ptr<rt::Runtime>w=c->owner;c->owner->syncStream();delete c;*expired=w.expired();return *expired?0:1;}catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int deep_hc_epilogue_hit_r358(void*raw,unsigned*out){auto*c=static_cast<Context*>(raw);try{owned(c);if(!out)throw std::invalid_argument("nullhitoutput");c->owner->memcpyD2H(out,c->hit->data(),4);c->owner->syncStream();return 0;}catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int deep_hc_epilogue_state_r358(void*raw,void*const*p){auto*c=static_cast<Context*>(raw);try{owned(c);if(!p)throw std::invalid_argument("stateoutputs");for(int i=0;i<7;++i)if(!p[i])throw std::invalid_argument("nullstateoutput");auto old=c->original->state_dict();rt::Tensor original[]={old.at("fn").tensor(),old.at("base").tensor(),old.at("scale").tensor()};rt::Tensor current[]={c->fn_,c->base_,c->scale_};for(int i=0;i<3;++i){c->owner->memcpyD2H(p[i],original[i]->data(),original[i]->nbytes());c->owner->memcpyD2H(p[i+3],current[i]->data(),current[i]->nbytes());}c->owner->memcpyD2H(p[6],c->constants_->data(),c->constants_->nbytes());c->owner->syncStream();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
