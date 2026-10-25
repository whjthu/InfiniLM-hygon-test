#pragma once
#include "runtime/runtime.hpp"
#include <cstdint>
#include <vector>
#include "batch_transaction_r9122.hpp"
namespace kimi_batch_r9122 {
namespace rt=infinilm::runtime;
using Request=kimi_batch_logic_r9122::Request;
void begin(rt::Runtime&,uint64_t,const std::vector<Request>&,size_t);
void clear(rt::Runtime&);
void capture(rt::Runtime&,size_t,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,rt::Tensor,float,float);
void replay(rt::Runtime&,uint64_t,const std::vector<size_t>&,rt::Tensor,rt::Tensor);
void stats(rt::Runtime&,uint64_t*,size_t);
}
#ifdef KIMI_STATE_ONLY_IMPLEMENTATION_R9113
#include <unordered_map>
#include <array>
#include <optional>
#include <set>
#include <algorithm>
namespace kimi_batch_r9122 {
struct Cached {uint64_t epoch=0;size_t total=0;bool armed=false,consumed=false;std::vector<Request>requests;rt::Tensor read,write;std::vector<int32_t>read_host,write_host;std::unordered_map<size_t,std::array<rt::Tensor,9>>layers;std::unordered_map<size_t,std::array<float,2>>scalars;std::array<uint64_t,8>counts{};};
static thread_local std::unordered_map<rt::Runtime*,Cached> states;
void clear(rt::Runtime&r){r.assertOwnerThread();r.syncStream();states.erase(&r);}
void begin(rt::Runtime&r,uint64_t epoch,const std::vector<Request>&req,size_t total){
 r.assertOwnerThread();if(r.isGraphRecording()||!epoch||req.empty()||req.size()>8||!total||total>16)throw std::invalid_argument("9122 bounded batch epoch");
 std::set<int>logical,rows;size_t offset=0;
 for(auto x:req){if(x.logical<1||x.logical>8||!logical.insert(x.logical).second||!((x.read==x.logical&&x.write==x.logical+8)||(x.write==x.logical&&x.read==x.logical+8))||!rows.insert(x.read).second||!rows.insert(x.write).second||x.past<1||x.offset!=offset||!x.width||x.width>8||size_t(x.past)+x.width>1024)throw std::invalid_argument("9122 disjoint per-request pair/past/offset");offset+=x.width;}
 if(offset!=total)throw std::invalid_argument("9122 total packed width");r.syncStream();auto&c=states[&r];c.layers.clear();c.scalars.clear();c.read={};c.write={};c.read_host.clear();c.write_host.clear();c.requests=req;c.epoch=epoch;c.total=total;c.armed=true;c.consumed=false;++c.counts[0];c.counts[7]+=req.size();
 for(auto&x:c.requests){c.read_host.push_back(x.read);c.write_host.push_back(x.write);}c.read=rt::Tensor::empty(r,{req.size()},rt::DataType::kInt32);c.write=rt::Tensor::empty(r,{req.size()},rt::DataType::kInt32);r.memcpyH2D(c.read->data(),c.read_host.data(),req.size()*sizeof(int32_t));r.memcpyH2D(c.write->data(),c.write_host.data(),req.size()*sizeof(int32_t));
}
void capture(rt::Runtime&r,size_t layer,rt::Tensor pre,rt::Tensor silu,rt::Tensor gate,rt::Tensor beta,rt::Tensor weight,rt::Tensor conv,rt::Tensor ssm,rt::Tensor alog,rt::Tensor bias,rt::Tensor read,rt::Tensor write,float scale,float lower){
 auto it=states.find(&r);if(it==states.end()||!it->second.armed)return;r.assertOwnerThread();auto&c=it->second;
 if(c.consumed||pre->shape()!=rt::Shape{1,c.total,2304}||silu->shape()!=pre->shape()||gate->shape()!=rt::Shape{1,c.total,6,128}||beta->shape()!=rt::Shape{1,c.total,6}||conv->shape()!=rt::Shape{17,2304,3}||ssm->shape()!=rt::Shape{17,6,128,128}||pre->dtype()!=rt::DataType::kBFloat16||read->numel()!=c.requests.size()||write->numel()!=c.requests.size()||!pre->is_contiguous()||!silu->is_contiguous()||!gate->is_contiguous()||!beta->is_contiguous()||c.layers.count(layer))throw std::invalid_argument("9122 actual packed batch capture geometry");
 c.layers[layer]={pre,silu,gate,beta,weight,conv,ssm,alog,bias};c.scalars[layer]={scale,lower};++c.counts[1];c.counts[6]+=pre->nbytes()+silu->nbytes()+gate->nbytes()+beta->nbytes();
}
void replay(rt::Runtime&r,uint64_t epoch,const std::vector<size_t>&retained,rt::Tensor conv,rt::Tensor ssm){
 r.assertOwnerThread();auto it=states.find(&r);if(it==states.end())throw std::invalid_argument("9122 missing owner");auto&c=it->second;
 if(!c.armed||c.consumed||c.epoch!=epoch||retained.size()!=c.requests.size()||c.layers.empty())throw std::invalid_argument("9122 stale/consumed batch epoch");
 for(size_t i=0;i<retained.size();++i)if(!retained[i]||retained[i]>c.requests[i].width)throw std::invalid_argument("9122 invalid per-request retention");
 for(auto&item:c.layers){auto&t=item.second;if(t[5]->data()!=conv->data()||t[6]->data()!=ssm->data()||conv->device()!=r.device()||ssm->device()!=r.device())throw std::invalid_argument("9122 wrong state pool owner");}
 struct Drain{rt::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{r};
 std::vector<size_t>selected;std::vector<int32_t>off{0},rr,ww;size_t tokens=0;
 for(size_t i=0;i<c.requests.size();++i)if(retained[i]<c.requests[i].width){selected.push_back(i);tokens+=retained[i];off.push_back(int32_t(tokens));rr.push_back(c.requests[i].read);ww.push_back(c.requests[i].write);}
 if(!selected.empty()){
  auto offsets=rt::Tensor::empty(r,{off.size()},rt::DataType::kInt32),reads=rt::Tensor::empty(r,{rr.size()},rt::DataType::kInt32),writes=rt::Tensor::empty(r,{ww.size()},rt::DataType::kInt32);r.memcpyH2D(offsets->data(),off.data(),off.size()*4);r.memcpyH2D(reads->data(),rr.data(),rr.size()*4);r.memcpyH2D(writes->data(),ww.data(),ww.size()*4);
  for(auto&item:c.layers){auto&t=item.second;auto sc=c.scalars.at(item.first);std::array<rt::Tensor,4>packed;
   for(size_t z=0;z<4;++z){auto shape=t[z]->shape();shape[1]=tokens;packed[z]=rt::Tensor::empty(r,shape,t[z]->dtype());size_t start=0;for(auto i:selected){auto n=retained[i];packed[z]->narrow({{1,start,n}})->copy_from(r,t[z]->narrow({{1,c.requests[i].offset,n}}));start+=n;}}
   auto pre=packed[0],silu=packed[1],gate=packed[2],beta=packed[3];auto convout=rt::Tensor::empty(r,pre->shape(),pre->dtype());const std::optional<rt::Tensor>none;auto cc=infini::ops::CausalConv1d::DefaultConfig(r.device().type());cc.set_implementation_index(0);
   infinilm::execution::submit<infini::ops::CausalConv1d>(r,infinilm::execution::RecordingPolicy::HostOnly,cc,pre,t[4],conv,none,std::optional<rt::Tensor>{offsets},std::optional<rt::Tensor>{reads},std::optional<rt::Tensor>{writes},convout,none);
   auto q=silu->narrow({{2,0,768}})->view({1,tokens,6,128}),k=silu->narrow({{2,768,768}})->view({1,tokens,6,128}),v=silu->narrow({{2,1536,768}})->view({1,tokens,6,128});auto out=rt::Tensor::empty(r,{1,tokens,6,128},rt::DataType::kBFloat16);auto kc=infini::ops::IndexedKimiDeltaAttention::DefaultConfig(r.device().type());kc.set_implementation_index(0);
   infinilm::execution::submit<infini::ops::IndexedKimiDeltaAttention>(r,infinilm::execution::RecordingPolicy::HostOnly,kc,out,ssm,q,k,v,gate,beta,t[7],t[8],offsets,reads,writes,sc[0],sc[1],true);r.syncStream();c.counts[2]+=selected.size();++c.counts[3];++c.counts[4];c.counts[5]+=tokens;
  }
  r.syncStream();
 }
 r.syncStream();c.layers.clear();c.scalars.clear();c.armed=false;c.consumed=true;
}

void stats(rt::Runtime&r,uint64_t*out,size_t n){r.assertOwnerThread();if(!out||n!=8)throw std::invalid_argument("9122 stats8");auto it=states.find(&r);for(size_t i=0;i<n;++i)out[i]=it==states.end()?0:it->second.counts[i];}
}
#endif
