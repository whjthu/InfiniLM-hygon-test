// Component boundary of current Math0 delta attention. No model, weights or CCL.
#include "execution/submit.hpp"
#include "base/causal_conv1d.h"
#include "base/indexed_kimi_delta_attention.h"
#include "base/silu.h"
#include <infini/rt/device_guard.h>
#include <cmath>
namespace rt=infinilm::runtime; namespace ex=infinilm::execution;
namespace { thread_local std::string error; struct Context {std::shared_ptr<rt::Runtime> r;uint64_t calls[4]{};}; }
extern "C" const char* kda_conv_error(){return error.c_str();}
extern "C" void* kda_conv_create(int device){try{
 if(device<0||device>=8)throw std::invalid_argument("local owner device");
 infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};
 auto c=new Context{rt::Runtime::create({rt::Device::Type::kHygon,device})};
 c->r->enableOwnerDrainFailStop();return c;
}catch(const std::exception&e){error=e.what();return nullptr;}}
// All operands GPU, same owner, BF16 data, F32 weights/a/bias, I32 metadata.
// p = projected_qkv,conv_weight,conv_pool,gate,beta,a_log,dt_bias,kda_pool,
//     offsets,reads,writes,conv_output,kda_output. Pools are disjoint-slot paired.
extern "C" int kda_conv_run(void* context,size_t T,size_t P,void**p){
 auto*c=static_cast<Context*>(context);try{
 if(!c||!p||!T||T>8||P<3||P>129)throw std::invalid_argument("bounded component geometry");
 auto&r=*c->r;r.assertOwnerThread();
 if(r.isGraphRecording())throw std::invalid_argument("eager only");
 constexpr size_t H=6,D=128,C=3*H*D;
 auto bf=rt::DataType::kBFloat16,f=rt::DataType::kFloat32,i=rt::DataType::kInt32;
 auto wrap=[&](size_t n,rt::Shape shape,rt::DataType dt){if(!p[n])throw std::invalid_argument("null operand");return rt::Tensor::from_blob(p[n],shape,dt,r.device());};
 auto x=wrap(0,{1,T,C},bf),w=wrap(1,{C,1,4},f),conv=wrap(2,{P,C,3},bf);
 auto gate=wrap(3,{1,T,H,D},bf),beta=wrap(4,{1,T,H},bf),a=wrap(5,{H},f),bias=wrap(6,{H,D},f),ssm=wrap(7,{P,H,D,D},bf);
 auto off=wrap(8,{2},i),read=wrap(9,{1},i),write=wrap(10,{1},i),y=wrap(11,{1,T,C},bf),out=wrap(12,{1,T,H,D},bf);
 std::optional<rt::Tensor> none;
 auto cc=infini::ops::CausalConv1d::DefaultConfig(r.device().type());cc.set_implementation_index(0);
 ex::submit<infini::ops::CausalConv1d>(r,ex::RecordingPolicy::HostOnly,cc,x,w,conv,none,std::optional<rt::Tensor>{off},std::optional<rt::Tensor>{read},std::optional<rt::Tensor>{write},y,none);
 ++c->calls[0];
 auto silu=rt::Tensor::empty(r,{1,T,C},bf);
 ex::submit<infini::ops::Silu>(r,infini::ops::Silu::DefaultConfig(r.device().type()),y,silu);
 ++c->calls[1];
 auto q=silu->narrow({{2,0,H*D}})->view({1,T,H,D});
 auto k=silu->narrow({{2,H*D,H*D}})->view({1,T,H,D});
 auto v=silu->narrow({{2,2*H*D,H*D}})->view({1,T,H,D});
 auto kc=infini::ops::IndexedKimiDeltaAttention::DefaultConfig(r.device().type());kc.set_implementation_index(0);
 ex::submit<infini::ops::IndexedKimiDeltaAttention>(r,ex::RecordingPolicy::HostOnly,kc,out,ssm,q,k,v,gate,beta,a,bias,off,read,write,1.f/std::sqrt(float(D)),-5.f,true);
 ++c->calls[2];c->calls[3]+=T;
 r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(c)try{c->r->syncStream();}catch(...){std::terminate();}return 1;}
}
extern "C" int kda_conv_counts(void*context,uint64_t*out){auto*c=static_cast<Context*>(context);if(!c||!out)return 1;try{c->r->assertOwnerThread();for(size_t i=0;i<4;++i)out[i]=c->calls[i];return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int kda_conv_close(void* context){auto*c=static_cast<Context*>(context);try{if(!c)return 1;c->r->assertOwnerThread();c->r->syncStream();c->r->trimMemory();delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
