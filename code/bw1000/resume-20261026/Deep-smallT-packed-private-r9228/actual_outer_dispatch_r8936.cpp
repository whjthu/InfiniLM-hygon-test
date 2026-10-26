#include <algorithm>
#include <infini/rt/device_guard.h>
#include <cstdint>
#include <cstdlib>
#include "deep_selection_r356.hpp"
#include "deepseek_v4_latent_attention.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <optional>
#include <limits>
#include <set>
#include <stdexcept>
#include <array>
#include <atomic>
extern "C" int deep_packed_gather_r374(void*,const void*,const void*,const void*,size_t,size_t,int64_t,int64_t,int64_t,int64_t,void*);
extern "C" int deep_latent_validate_r681(const void*,const void*,const void*,size_t,size_t,size_t,size_t,size_t,void*,void*);
extern "C" int deep_latent_compute_r681(void*,void*,size_t,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,size_t,size_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,void*);
namespace infinilm::models::deepseek_v4 {
runtime::Tensor latent_attention_reference_r681(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&);

namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8> fused_calls{},fallback_calls{};}
namespace {
std::atomic<int> prefill_tiles_mode{0};
std::array<std::array<std::atomic<uint64_t>,3>,8> prefill_tiles_counts{};
rt::Tensor prefill_tiles_r8933(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&swa,const rt::Tensor&compressed,const rt::Tensor&sink,const rt::Tensor&si,const rt::Tensor&ci,const rt::Tensor&lengths){
 infini::rt::DeviceGuard device_scope(r.device().type(),r.device().index(),true);
 const size_t tokens=q->size(0),heads=q->size(1),sw=si->size(1),cw=ci->size(1),parts=(sw+cw+64)/64;
 auto out=rt::Tensor::empty(r,q->shape(),q->dtype()),flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32),work=rt::Tensor::empty(r,{32,heads,parts,514},rt::DataType::kFloat32);
 try{
  for(size_t begin=0;begin<tokens;begin+=32){
   const size_t width=std::min(size_t{32},tokens-begin);if(width<8)throw std::logic_error("prefill tail must use original fallback");
   auto query=q->narrow({{0,begin,width}}),s=si->narrow({{0,begin,width}}),c=ci->narrow({{0,begin,width}}),lens=lengths->narrow({{0,begin,width}});
   int code=0;r.memcpyH2D(flag->data(),&code,sizeof(code),true);
   if(deep_latent_validate_r681(s->data(),c->data(),lens->data(),width,sw,cw,swa->size(0),compressed->size(0),flag->data(),r.stream()))throw std::runtime_error("prefill tile validate launch");
   r.memcpyD2H(&code,flag->data(),sizeof(code));if(code)throw std::invalid_argument("prefill tile invalid selection: "+std::to_string(code));
   auto destination=out->data()+begin*heads*512*sizeof(uint16_t);
   if(deep_latent_compute_r681(destination,work->data(),parts,query->data(),swa->data(),compressed->data(),sink->data(),s->data(),c->data(),lens->data(),width,heads,sw,cw,query->stride(0),query->stride(1),query->stride(2),swa->stride(0),swa->stride(1),compressed->stride(0),compressed->stride(1),sink->stride(0),r.stream()))throw std::runtime_error("prefill tile compute launch");
   // Same completion fence as one original fused call; views/workspace remain live to here.
   r.syncStream();++prefill_tiles_counts.at(r.device().index())[2];
  }
  auto&counts=prefill_tiles_counts.at(r.device().index());++counts[0];counts[1].fetch_add(tokens);return out;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
rt::Tensor latent_attention(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&swa,const rt::Tensor&compressed,const rt::Tensor&sink,const rt::Tensor&si,const rt::Tensor&ci,const rt::Tensor&lengths){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!swa||!compressed||!sink||!si||!ci||!lengths||q->ndim()!=3||!q->size(0)||!q->size(1)||q->size(2)!=512||q->dtype()!=rt::DataType::kBFloat16||swa->ndim()!=2||compressed->ndim()!=2||swa->size(1)!=512||compressed->size(1)!=512||!swa->size(0)||!compressed->size(0)||swa->dtype()!=q->dtype()||compressed->dtype()!=q->dtype()||sink->shape()!=rt::Shape{q->size(1)}||sink->dtype()!=rt::DataType::kFloat32||si->ndim()!=2||ci->ndim()!=2||si->size(0)!=q->size(0)||ci->size(0)!=q->size(0)||!si->size(1)||!ci->size(1)||si->dtype()!=rt::DataType::kInt64||ci->dtype()!=rt::DataType::kInt64||lengths->shape()!=rt::Shape{q->size(0),2}||lengths->dtype()!=rt::DataType::kInt32)
  throw std::invalid_argument("DeepSeek latent attention requires BF16 D512 KV, F32 sink and explicit selected lengths");
 for(const auto&x:{q,swa,compressed,sink,si,ci,lengths})if(x->device()!=r.device())throw std::invalid_argument("DeepSeek latent attention owner differs");
 if(!si->is_contiguous()||!ci->is_contiguous()||!lengths->is_contiguous())throw std::invalid_argument("DeepSeek selection metadata must be contiguous");
 const size_t tokens=q->size(0),heads=q->size(1),sw=si->size(1),cw=ci->size(1);
 if(prefill_tiles_mode.load()==1&&tokens>64&&tokens<=4096&&(tokens%32==0||tokens%32>=8)&&heads<=128&&sw<=2048&&cw<=2048-sw&&swa->size(0)<=size_t(INT64_MAX)&&compressed->size(0)<=size_t(INT64_MAX)-swa->size(0))return prefill_tiles_r8933(r,q,swa,compressed,sink,si,ci,lengths);
 if(tokens<8||tokens>64||heads>128||sw>2048||cw>2048-sw||swa->size(0)>size_t(INT64_MAX)||compressed->size(0)>size_t(INT64_MAX)-swa->size(0)){auto result=latent_attention_reference_r681(r,q,swa,compressed,sink,si,ci,lengths);fallback_calls.at(r.device().index()).fetch_add(1);return result;}
 try{
  auto flag=rt::Tensor::empty(r,{1},rt::DataType::kInt32);int code=0;r.memcpyH2D(flag->data(),&code,sizeof(code),true);
  if(deep_latent_validate_r681(si->data(),ci->data(),lengths->data(),tokens,sw,cw,swa->size(0),compressed->size(0),flag->data(),r.stream()))throw std::runtime_error("Deep fused selection validation launch failed");
  r.memcpyD2H(&code,flag->data(),sizeof(code));
  if(code)throw std::invalid_argument("Deep fused invalid selection lengths, range or duplicates: "+std::to_string(code));
  auto out=rt::Tensor::empty(r,q->shape(),q->dtype());size_t parts=(sw+cw+64)/64;auto work=rt::Tensor::empty(r,{tokens,heads,parts,514},rt::DataType::kFloat32);
  if(deep_latent_compute_r681(out->data(),work->data(),parts,q->data(),swa->data(),compressed->data(),sink->data(),si->data(),ci->data(),lengths->data(),tokens,heads,sw,cw,q->stride(0),q->stride(1),q->stride(2),swa->stride(0),swa->stride(1),compressed->stride(0),compressed->stride(1),sink->stride(0),r.stream()))throw std::runtime_error("Deep fused attention launch failed");
  r.syncStream();fused_calls.at(r.device().index()).fetch_add(1);return out;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
extern "C" uint64_t modern_deep_latent_fused_calls_r681(int d){return d>=0&&d<8?infinilm::models::deepseek_v4::fused_calls[size_t(d)].load():0;}
extern "C" uint64_t modern_deep_latent_fallback_calls_r681(int d){return d>=0&&d<8?infinilm::models::deepseek_v4::fallback_calls[size_t(d)].load():0;}

extern "C" int deep_latent_prefill_mode_r8933(int mode){if(mode!=0&&mode!=1)return 1;infinilm::models::deepseek_v4::prefill_tiles_mode.store(mode);return 0;}
extern "C" int deep_latent_prefill_counts_r8933(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<n;++i)out[i]=infinilm::models::deepseek_v4::prefill_tiles_counts[d][i].load();return 0;}

#include "deepseek_v4_latent_attention.hpp"
#include <cuda_runtime_api.h>
#include "execution/submit.hpp"
#include <optional>
extern "C" int deep_packed_gather_r374(void*,const void*,const void*,const void*,size_t,size_t,int64_t,int64_t,int64_t,int64_t,void*);
#include <map>
#include <infini/rt/device_guard.h>
#include <memory>
#include <string>
namespace rt=infinilm::runtime;
namespace{thread_local std::string error;std::map<int,std::shared_ptr<rt::Runtime>>&owners(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>v;return v;}std::shared_ptr<rt::Runtime>worker(int d){auto&r=owners()[d];if(!r){r=rt::Runtime::create({rt::Device::Type::kHygon,d});r->enableOwnerDrainFailStop();}return r;}}
extern "C" const char*modern_deep_latent_error(){return error.c_str();}
extern "C" int modern_deep_latent(int device,size_t tokens,size_t heads,size_t swa_rows,size_t compressed_rows,size_t sw,size_t cw,const void*q,const void*swa,const void*compressed,const void*sink,const void*si,const void*ci,const void*lengths,void*out,void*stream){std::shared_ptr<rt::Runtime>owner;try{
 infini::rt::DeviceGuard fixture_guard{rt::Device::Type::kHygon,device,true};
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("DeepSeek latent producer stream failed");owner=worker(device);auto&r=*owner;auto view=[&](const void*p,rt::Shape shape,rt::DataType dt){return rt::Tensor::from_blob(const_cast<void*>(p),shape,dt,r.device());};
 auto output=infinilm::models::deepseek_v4::latent_attention(r,view(q,{tokens,heads,512},rt::DataType::kBFloat16),view(swa,{swa_rows,512},rt::DataType::kBFloat16),view(compressed,{compressed_rows,512},rt::DataType::kBFloat16),view(sink,{heads},rt::DataType::kFloat32),view(si,{tokens,sw},rt::DataType::kInt64),view(ci,{tokens,cw},rt::DataType::kInt64),view(lengths,{tokens,2},rt::DataType::kInt32));view(out,{tokens,heads,512},rt::DataType::kBFloat16)->copy_from(r,output);r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(owner)try{owner->syncStream();}catch(...){std::terminate();}return 1;}}

extern "C" int modern_deep_latent_strided_r681(int device,size_t tokens,size_t heads,size_t swa_rows,size_t compressed_rows,size_t sw,size_t cw,int64_t q0,int64_t q1,int64_t q2,int64_t s0,int64_t s1,int64_t c0,int64_t c1,int64_t ss,const void*q,const void*swa,const void*compressed,const void*sink,const void*si,const void*ci,const void*lengths,void*out,void*stream){std::shared_ptr<rt::Runtime>owner;try{
 infini::rt::DeviceGuard fixture_guard{rt::Device::Type::kHygon,device,true};
 if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("DeepSeek latent producer stream failed");owner=worker(device);auto&r=*owner;auto view=[&](const void*p,rt::Shape shape,rt::DataType dt){return rt::Tensor::from_blob(const_cast<void*>(p),shape,dt,r.device());};
 auto strided=[&](const void*p,rt::Shape shape,rt::Strides strides,rt::DataType dt){return rt::Tensor::strided_from_blob(const_cast<void*>(p),shape,strides,dt,r.device());};
 auto output=infinilm::models::deepseek_v4::latent_attention(r,strided(q,{tokens,heads,512},{q0,q1,q2},rt::DataType::kBFloat16),strided(swa,{swa_rows,512},{s0,s1},rt::DataType::kBFloat16),strided(compressed,{compressed_rows,512},{c0,c1},rt::DataType::kBFloat16),strided(sink,{heads},{ss},rt::DataType::kFloat32),view(si,{tokens,sw},rt::DataType::kInt64),view(ci,{tokens,cw},rt::DataType::kInt64),view(lengths,{tokens,2},rt::DataType::kInt32));view(out,{tokens,heads,512},rt::DataType::kBFloat16)->copy_from(r,output);r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(owner)try{owner->syncStream();}catch(...){std::terminate();}return 1;}}

extern "C" int modern_deep_latent_shutdown(){try{for(auto&item:owners()){infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,item.first,true};item.second->syncStream();item.second->trimMemory();item.second.reset();}owners().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
