// Private component only: no weights, Engine, communicator or Model promotion.
#include "glm_moe_dsa_sparse_attention.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <set>
#include <optional>
#include <stdexcept>
#include <vector>
namespace rt=infinilm::runtime;
namespace ex=infinilm::execution;
namespace model=infinilm::models::glm_moe_dsa;
extern "C" int modern_glm_dense_mode_r48_set(int);
extern "C" int modern_glm_dense_causal_mask_r25(void*,size_t,size_t,size_t,size_t,void*);
namespace {
constexpr size_t B=8,L=512,H=8,D=256;
thread_local std::string error;
struct Context {std::shared_ptr<rt::Runtime> r;int device;};
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void owner(Context*c){if(!c||!c->r)throw std::invalid_argument("null context");c->r->assertOwnerThread();int d=-1;ck(cudaGetDevice(&d));if(d!=c->device||c->r->isGraphRecording())throw std::invalid_argument("owner/device/eager");}
void pointer(Context*c,const void*p){if(!p)throw std::invalid_argument("null pointer");cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device)throw std::invalid_argument("device pointer");}
struct Work {rt::Runtime&r;std::vector<rt::Tensor> held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto t=rt::Tensor::empty(r,s,d);held.push_back(t);return t;}};
void gemm(rt::Runtime&r,const rt::Tensor&a,const rt::Tensor&b,const rt::Tensor&c){ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),a,b,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},c);}
void cast(rt::Runtime&r,const rt::Tensor&a,const rt::Tensor&b){ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),a,b);}
rt::Tensor row(const rt::Tensor&t,size_t b){return t->narrow({{0,b,1}})->view({L,H,D});}
// Match old full-prefix set/causality checks; validation remains inside wall.
void validate(rt::Runtime&r,const rt::Tensor&ids,const rt::Tensor&positions){
 std::vector<int64_t>pos(L),host(B*L*L);r.memcpyD2H(pos.data(),positions->data(),L*8);r.memcpyD2H(host.data(),ids->data(),host.size()*8);
 for(size_t i=0;i<L;++i)if(pos[i]!=int64_t(i))throw std::invalid_argument("nonzero/noncanonical prefix");
 for(size_t b=0;b<B;++b)for(size_t t=0;t<L;++t){std::set<int64_t> selected;for(size_t j=0;j<L;++j){auto id=host[(b*L+t)*L+j];if(id<0||size_t(id)>=L)throw std::invalid_argument("selection range");if(size_t(id)<=t)selected.insert(id);}if(selected.size()!=t+1)throw std::invalid_argument("not complete causal selection");size_t j=0;for(auto id:selected)if(id!=int64_t(j++))throw std::invalid_argument("not source ordered");}
}
rt::Tensor candidate(Work&w,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&ids,const rt::Tensor&pos){
 auto&r=w.r;validate(r,ids,pos);
 // Pack the same eight independent requests; no token/feature padding.
 auto qp=q->permute({0,2,1,3})->contiguous(r)->view({B*H,L,D});
 auto kp=k->permute({0,2,1,3})->contiguous(r)->view({B*H,L,D});
 auto vp=v->permute({0,2,1,3})->contiguous(r)->view({B*H,L,D});
 w.held.insert(w.held.end(),{qp,kp,vp});
 auto dots=w.alloc({B*H,L,L},rt::DataType::kBFloat16);r.setDeviceMemory(dots->data(),0,dots->nbytes());
 auto temp=w.alloc({B*H,L},rt::DataType::kBFloat16);
 for(size_t t=0;t<L;++t){size_t count=t+1;auto out=rt::Tensor::from_blob(temp->data(),{B*H,1,count},rt::DataType::kBFloat16,r.device());
  gemm(r,qp->narrow({{1,t,1}}),kp->narrow({{1,0,count}})->permute({0,2,1}),out);
  dots->narrow({{1,t,1},{2,0,count}})->copy_from(r,out);
 }
 // Preserve mode2's F32 scaling then BF16 rounding, F32 logits, Softmax, BF16 P.
 auto df=w.alloc(dots->shape(),rt::DataType::kFloat32),scaled=w.alloc(dots->shape(),rt::DataType::kFloat32),rounded=w.alloc(dots->shape(),rt::DataType::kBFloat16),logits=w.alloc(dots->shape(),rt::DataType::kFloat32),prob=w.alloc(dots->shape(),rt::DataType::kFloat32),pb=w.alloc(dots->shape(),rt::DataType::kBFloat16);
 float scale=1.f/16;auto sf=w.alloc({1},rt::DataType::kFloat32);r.memcpyH2D(sf->data(),&scale,4,false);cast(r,dots,df);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),df,sf,scaled);cast(r,scaled,rounded);cast(r,rounded,logits);
 if(modern_glm_dense_causal_mask_r25(logits->data(),B*H,L,L,0,r.stream()))throw std::runtime_error("mask launch");
 r.setDeviceMemory(prob->data(),0,prob->nbytes());r.setDeviceMemory(pb->data(),0,pb->nbytes());
 ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},prob);cast(r,prob,pb);
 auto output=w.alloc({B*H,L,D},rt::DataType::kBFloat16),ptmp=w.alloc({B*H,L},rt::DataType::kBFloat16);
 for(size_t t=0;t<L;++t){size_t count=t+1;auto p=rt::Tensor::from_blob(ptmp->data(),{B*H,1,count},rt::DataType::kBFloat16,r.device());p->copy_from(r,pb->narrow({{1,t,1},{2,0,count}}));auto out=output->narrow({{1,t,1}});gemm(r,p,vp->narrow({{1,0,count}}),out);}
 auto packed=output->view({B,H,L,D})->permute({0,2,1,3})->contiguous(r);w.held.push_back(packed);return packed;
}
}
extern "C" const char*glm_crossrequest_error_r7314(){return error.c_str();}
extern "C" void*glm_crossrequest_create_r7314(int device){try{if(device<0||device>7)throw std::invalid_argument("device");infini::rt::DeviceGuard g{rt::Device::Type::kHygon,device,true};auto c=std::make_unique<Context>();c->device=device;c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();return c.release();}catch(const std::exception&e){error=e.what();return nullptr;}}
// q/k/v and output: contiguous BF16 [8,512,8,256]; ids I64 [8,512,512]; pos I64[512].
// Caller completes producer before entry. seconds[0] is complete CAPI wall incl drain.
extern "C" int glm_crossrequest_forward_r7314(void*raw,int on,const void*q,const void*k,const void*v,const void*ids,const void*pos,void*out,double*seconds){auto*c=static_cast<Context*>(raw);try{owner(c);if((on!=0&&on!=1)||!seconds)throw std::invalid_argument("mode/output");for(auto p:{q,k,v,ids,pos,static_cast<const void*>(out)})pointer(c,p);auto&r=*c->r;Work w{r};auto wrap=[&](const void*p,rt::Shape s,rt::DataType d){auto t=rt::Tensor::from_blob(const_cast<void*>(p),s,d,r.device());w.held.push_back(t);return t;};auto Q=wrap(q,{B,L,H,D},rt::DataType::kBFloat16),K=wrap(k,{B,L,H,D},rt::DataType::kBFloat16),V=wrap(v,{B,L,H,D},rt::DataType::kBFloat16),I=wrap(ids,{B,L,L},rt::DataType::kInt64),P=wrap(pos,{L},rt::DataType::kInt64),O=wrap(out,{B,L,H,D},rt::DataType::kBFloat16);r.syncStream();auto start=std::chrono::steady_clock::now();
 if(on)O->copy_from(r,candidate(w,Q,K,V,I,P));
 else{if(modern_glm_dense_mode_r48_set(2))throw std::runtime_error("old mode2");for(size_t b=0;b<B;++b){auto ir=I->narrow({{0,b,1}})->view({L,L});auto result=model::sparse_attention_mapped_bounded(r,row(Q,b),row(K,b),row(V,b),ir,P,0,1.f/16);w.held.push_back(result);row(O,b)->copy_from(r,result);}}
 r.syncStream();seconds[0]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return 0;
 }catch(const std::exception&e){error=e.what();if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int glm_crossrequest_destroy_r7314(void*raw,int*expired){auto*c=static_cast<Context*>(raw);try{owner(c);if(!expired)throw std::invalid_argument("expired");std::weak_ptr<rt::Runtime>w=c->r;c->r->syncStream();c->r->trimMemory();c->r.reset();*expired=w.expired();delete c;return *expired?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
