#pragma once
#include "runtime/runtime.hpp"
#include <cstdint>
namespace kimi_state_only_r9113 {
namespace rt=infinilm::runtime;
__attribute__((visibility("default"))) void begin(rt::Runtime&,std::uint64_t,int,int,size_t);
__attribute__((visibility("default"))) void invalidate(rt::Runtime&);
__attribute__((visibility("default"))) void clear(rt::Runtime&);
__attribute__((visibility("default"))) void capture(rt::Runtime&,size_t,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,float,float);
__attribute__((visibility("default"))) void replay(rt::Runtime&,size_t,std::uint64_t,size_t,rt::Tensor,rt::Tensor);
__attribute__((visibility("default"))) void stats(rt::Runtime&,std::uint64_t*,size_t);
}
#ifdef KIMI_STATE_ONLY_IMPLEMENTATION_R9113
#include <unordered_map>
#include <array>
#include <optional>
namespace kimi_state_only_r9113 {
struct Cached {std::uint64_t epoch=0;int read=0,write=0;size_t width=0;bool armed=false,consumed=false;std::unordered_map<size_t,std::array<rt::Tensor,11>> layers;std::unordered_map<size_t,std::array<float,2>> scalars;std::array<std::uint64_t,7> counts{};};
static thread_local std::unordered_map<rt::Runtime*,Cached> state;
void begin(rt::Runtime&r,std::uint64_t epoch,int read,int write,size_t width){r.assertOwnerThread();if(r.isGraphRecording()||!epoch||!width||width>8||(read!=1&&read!=9)||(write!=1&&write!=9)||read==write)throw std::invalid_argument("r9113 bounded paired verify epoch");auto&c=state[&r];c.layers.clear();c.scalars.clear();c.epoch=epoch;c.read=read;c.write=write;c.width=width;c.armed=true;c.consumed=false;++c.counts[0];}
void invalidate(rt::Runtime&r){r.assertOwnerThread();auto it=state.find(&r);if(it!=state.end()){it->second.layers.clear();it->second.scalars.clear();it->second.armed=false;it->second.consumed=true;}}
void clear(rt::Runtime&r){r.assertOwnerThread();r.syncStream();state.erase(&r);}
void capture(rt::Runtime&r,size_t layer,rt::Tensor pre,rt::Tensor silu,rt::Tensor gate,rt::Tensor beta,rt::Tensor weight,rt::Tensor conv,rt::Tensor ssm,rt::Tensor alog,rt::Tensor bias,rt::Tensor read,rt::Tensor write,float scale,float lower){
 auto it=state.find(&r);if(it==state.end()||!it->second.armed)return;auto&c=it->second;r.assertOwnerThread();
 if(c.consumed||pre->shape()!=rt::Shape{1,c.width,2304}||silu->shape()!=pre->shape()||gate->shape()!=rt::Shape{1,c.width,6,128}||beta->shape()!=rt::Shape{1,c.width,6}||conv->shape()!=rt::Shape{17,2304,3}||ssm->shape()!=rt::Shape{17,6,128,128}||pre->dtype()!=rt::DataType::kBFloat16||read->numel()!=1||write->numel()!=1||!pre->is_contiguous()||!silu->is_contiguous()||!gate->is_contiguous()||!beta->is_contiguous())throw std::invalid_argument("r9113 actual H6 packed capture contract");
 // All producer outputs are fresh allocating tensors. Original Conv/KDA read
 // inputs as const and subsequent norm/o_proj consume separate KDA output.
 // Keep allocations alive until replay; no CPU/raw tensor capture and no clone.
 if(c.layers.count(layer))throw std::invalid_argument("r9113 duplicate layer within verify epoch");
 c.layers.emplace(layer,std::array<rt::Tensor,11>{pre,silu,gate,beta,weight,conv,ssm,alog,bias,read,write});c.scalars.emplace(layer,std::array<float,2>{scale,lower});++c.counts[1];c.counts[6]+=pre->nbytes()+silu->nbytes()+gate->nbytes()+beta->nbytes();
}
void replay(rt::Runtime&r,size_t layer,std::uint64_t epoch,size_t retained,rt::Tensor expected_conv,rt::Tensor expected_ssm){
 r.assertOwnerThread();auto it=state.find(&r);if(it==state.end())throw std::invalid_argument("r9113 no cache owner");auto&c=it->second;
 if(!c.armed||c.consumed||epoch!=c.epoch||!retained||retained>c.width||!c.layers.count(layer))throw std::invalid_argument("r9113 stale epoch/width/cache");auto&t=c.layers.at(layer);auto sc=c.scalars.at(layer);
 if(expected_conv->data()!=t[5]->data()||expected_ssm->data()!=t[6]->data()||t[5]->device()!=r.device()||t[6]->device()!=r.device())throw std::invalid_argument("r9113 wrong state owner/pool");
 std::vector<rt::Tensor>held;held.reserve(3);struct Drain{rt::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{r};
 auto prefix=[&](const rt::Tensor&x){return x->narrow({{1,0,retained}});};auto pre=prefix(t[0]),silu=prefix(t[1]),gate=prefix(t[2]),beta=prefix(t[3]);
 int32_t off[2]={0,int32_t(retained)};auto offsets=rt::Tensor::empty(r,{2},rt::DataType::kInt32);held.push_back(offsets);r.memcpyH2D(offsets->data(),off,sizeof(off));
 auto conv_out=rt::Tensor::empty(r,pre->shape(),pre->dtype());held.push_back(conv_out);const std::optional<rt::Tensor>none;
 auto cc=infini::ops::CausalConv1d::DefaultConfig(r.device().type());cc.set_implementation_index(0);
 infinilm::execution::submit<infini::ops::CausalConv1d>(r,infinilm::execution::RecordingPolicy::HostOnly,cc,pre,t[4],t[5],none,std::optional<rt::Tensor>{offsets},std::optional<rt::Tensor>{t[9]},std::optional<rt::Tensor>{t[10]},conv_out,none);++c.counts[3];
 auto q=silu->narrow({{2,0,768}})->view({1,retained,6,128});auto k=silu->narrow({{2,768,768}})->view({1,retained,6,128});auto v=silu->narrow({{2,1536,768}})->view({1,retained,6,128});auto out=rt::Tensor::empty(r,{1,retained,6,128},rt::DataType::kBFloat16);
 held.push_back(out);auto kc=infini::ops::IndexedKimiDeltaAttention::DefaultConfig(r.device().type());kc.set_implementation_index(0);
 infinilm::execution::submit<infini::ops::IndexedKimiDeltaAttention>(r,infinilm::execution::RecordingPolicy::HostOnly,kc,out,t[6],q,k,v,gate,beta,t[7],t[8],offsets,t[9],t[10],sc[0],sc[1],true);++c.counts[4];++c.counts[2];c.counts[5]+=retained;
 // Single layer fixture consumes one layer. Model integration must explicitly
 // finish all captured layers before advancing the outer pair transaction.
 r.syncStream();c.layers.erase(layer);c.scalars.erase(layer);if(c.layers.empty()){c.consumed=true;c.armed=false;}
}
void stats(rt::Runtime&r,std::uint64_t*out,size_t n){r.assertOwnerThread();if(!out||n!=7)throw std::invalid_argument("r9113 stats shape");auto it=state.find(&r);for(size_t i=0;i<n;++i)out[i]=it==state.end()?0:it->second.counts[i];}
}
#endif
