#include "runtime/runtime.hpp"
#include "layers/quantization/none_quantization.hpp"
#include <infini/rt/device_guard.h>
#include <array>
#include <vector>
#include <cstring>
#include <memory>
#include <string>
namespace rt=infinilm::runtime;namespace qt=infinilm::quantization;
namespace{thread_local std::string error;struct C{std::shared_ptr<rt::Runtime>r;rt::Tensor w;qt::ParamsMap p;std::shared_ptr<qt::NoneQuantization>q;std::array<uint64_t,3>counts{};std::vector<unsigned char>reference;size_t k=0,n=0;};void drain(C*c)noexcept{if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}}}
extern "C" const char*proj_error(){return error.c_str();}
extern "C" void*gate_proj_create_r9032(size_t k,size_t n,const void*w){std::unique_ptr<C>c;try{if(!w||!((k==7168&&(n==128||n==6||n==768))||(k==128&&n==768)))throw std::invalid_argument("bounded actual gate weight geometry");infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,0,true};c=std::make_unique<C>();c->k=k;c->n=n;c->r=rt::Runtime::create({rt::Device::Type::kHygon,0});c->r->enableOwnerDrainFailStop();c->q=std::make_shared<qt::NoneQuantization>();c->w=rt::Tensor::empty(*c->r,{n,k},rt::DataType::kBFloat16);auto h=rt::Tensor::from_blob(const_cast<void*>(w),{n,k},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});c->w->copy_from(*c->r,h);c->reference.assign(static_cast<const unsigned char*>(w),static_cast<const unsigned char*>(w)+n*k*2);c->p.emplace("weight",c->w);drain(c.get());return c.release();}catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" int proj_run(void*raw,size_t m,const void*x,void*y){auto*c=static_cast<C*>(raw);rt::Tensor host,input,out;try{if(!c||!x||!y||(m!=1&&m!=8))throw std::invalid_argument("M1/M8 only");auto&r=*c->r;r.assertOwnerThread();host=rt::Tensor::from_blob(const_cast<void*>(x),{1,m,c->k},rt::DataType::kBFloat16,{rt::Device::Type::kCpu,0});input=rt::Tensor::empty(r,{1,m,c->k},rt::DataType::kBFloat16);input->copy_from(r,host);out=c->q->forward(r,c->p,input,false,1.f);if(out->shape()!=rt::Shape{1,m,c->n}||out->dtype()!=rt::DataType::kBFloat16)throw std::logic_error("provider output geometry");r.syncStream();r.memcpyD2H(y,out->data(),out->nbytes());r.syncStream();++c->counts[m==1?0:1];c->counts[2]+=m;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int proj_counts(void*raw,uint64_t*out){auto*c=static_cast<C*>(raw);if(!c||!out)return 1;c->r->assertOwnerThread();for(size_t i=0;i<3;++i)out[i]=c->counts[i];return 0;}
extern "C" int proj_close(void*raw){auto*c=static_cast<C*>(raw);if(!c)return 1;drain(c);c->p.clear();c->w={};c->q.reset();drain(c);delete c;return 0;}

extern "C" int proj_readonly(void*raw){auto*c=static_cast<C*>(raw);try{if(!c)throw std::invalid_argument("nullcontext");std::vector<unsigned char>v(c->reference.size());drain(c);c->r->memcpyD2H(v.data(),c->w->data(),v.size());drain(c);if(v!=c->reference)throw std::runtime_error("GPU weight mutated");return 0;}catch(const std::exception&e){error=e.what();return 1;}}
