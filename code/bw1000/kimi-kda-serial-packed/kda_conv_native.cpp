// Component boundary of current Math0 delta attention. No model, weights or CCL.
#include "execution/submit.hpp"
#include "base/causal_conv1d.h"
#include "base/indexed_kimi_delta_attention.h"
#include "base/silu.h"
#include <infini/rt/device_guard.h>
#include <cmath>
#include <array>
#include <vector>
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
static int kda_conv_run_impl(void* context,size_t T,size_t P,void**p,bool serial_kda){
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
 // All views, metadata and the host H2D source survive until the stream drains.
 std::array<int32_t,2> one_offsets{{0,1}};
 rt::Tensor step_offsets;
 std::vector<rt::Tensor> token_views;
 std::vector<rt::Tensor> packed_operands;
 struct DrainBeforeViews {rt::Runtime&r;bool done=false;~DrainBeforeViews()noexcept{if(!done)try{r.syncStream();}catch(...){std::terminate();}}} drain{r};
 if(serial_kda){
  step_offsets=rt::Tensor::empty(r,{2},i);
  r.memcpyH2D(step_offsets->data(),one_offsets.data(),sizeof(one_offsets));
  token_views.reserve(T*6);
  packed_operands.reserve(T*6);
  for(size_t j=0;j<T;++j){
   const size_t base=token_views.size();
   for(const auto&t:{out,q,k,v,gate,beta})token_views.push_back(t->narrow({{1,j,1}}));
   packed_operands.push_back(rt::Tensor::empty(r,{1,1,H,D},bf));
   for(size_t n=1;n<6;++n)packed_operands.push_back(token_views[base+n]->contiguous(r));
   // First token copies active->shadow; subsequent T1 calls use ordinary in-place
   // state semantics. Each original Math0 call stores BF16 state before the next.
   auto rr=j==0?read:write;
   ex::submit<infini::ops::IndexedKimiDeltaAttention>(r,ex::RecordingPolicy::HostOnly,kc,
    packed_operands[base],ssm,packed_operands[base+1],packed_operands[base+2],packed_operands[base+3],
    packed_operands[base+4],packed_operands[base+5],a,bias,step_offsets,rr,write,
    1.f/std::sqrt(float(D)),-5.f,true);
   ++c->calls[2];
   token_views[base]->copy_from(r,packed_operands[base]);
  }
 }else{
  ex::submit<infini::ops::IndexedKimiDeltaAttention>(r,ex::RecordingPolicy::HostOnly,kc,out,ssm,q,k,v,gate,beta,a,bias,off,read,write,1.f/std::sqrt(float(D)),-5.f,true);
  ++c->calls[2];
 }
 c->calls[3]+=T;
 r.syncStream();drain.done=true;return 0;
 }catch(const std::exception&e){error=e.what();if(c)try{c->r->syncStream();}catch(...){std::terminate();}return 1;}
}
// Existing ABI stays Math0 chunk; the new explicit diagnostic entry splits KDA only.
extern "C" int kda_conv_run(void* c,size_t T,size_t P,void**p){return kda_conv_run_impl(c,T,P,p,false);}
extern "C" int kda_conv_run_serial_kda(void* c,size_t T,size_t P,void**p){return kda_conv_run_impl(c,T,P,p,true);}
extern "C" int kda_conv_counts(void*context,uint64_t*out){auto*c=static_cast<Context*>(context);if(!c||!out)return 1;try{c->r->assertOwnerThread();for(size_t i=0;i<4;++i)out[i]=c->calls[i];return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int kda_conv_close(void* context){auto*c=static_cast<Context*>(context);try{if(!c)return 1;c->r->assertOwnerThread();c->r->syncStream();c->r->trimMemory();delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
