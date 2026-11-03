#pragma once
// Diagnostic only. All GPU Tensor handles below belong to the real Runtime
// owner stack. The public CABI stores/reads CPU bytes only after existing role
// completion fences. No parameter, main output, class layout or SDK is changed.
#include "activation_quant.hpp"
#include "model_scope.hpp"
#include "execution/submit.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <vector>
namespace deep438_capture {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
constexpr size_t owners=8,layers=43,record_width=12,tensor_width=5;
constexpr size_t store_limit=16*1024*1024;
struct TensorBytes{std::array<uint64_t,tensor_width>info{};std::vector<unsigned char>bytes;};
struct Record{std::array<uint64_t,record_width>info{};std::vector<TensorBytes>fields;};
struct Owner{std::vector<Record>records;bool failed=false;};
inline std::array<Owner,owners>storage;
inline std::atomic<bool>armed{false},sealed{false};
inline std::atomic<uint64_t>pass{0},output{0},forward{0};
inline std::mutex control;
struct Entry{rt::Runtime&r;int owner;uint64_t layer;rt::Tensor hidden;unsigned finished=0;};
inline thread_local Entry*entry=nullptr;
inline rt::Tensor clone(rt::Runtime&r,const rt::Tensor&t){auto s=rt::Tensor::empty(r,t->shape(),t->dtype());try{s->copy_from(r,t);}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}return s;}
class Scope{
 rt::Runtime&r_;Entry state_;bool active_=false;
 public:
 Scope(rt::Runtime&r,const rt::Tensor&x):r_(r),state_{r,r.device().index(),0,{}}{
  if(!armed.load())return;r.assertOwnerThread();
  if(entry||!deep438::state.active||!deep438::state.t1||!x||x->shape()!=rt::Shape{1,4096}||x->dtype()!=rt::DataType::kBFloat16||!x->is_contiguous())throw std::logic_error("diagnostic Scope is not the actual eligible T1 Model entry");
  if(state_.owner<0||state_.owner>=8)throw std::logic_error("diagnostic owner");
  auto ordinal=deep438_model::rows[state_.owner].value[deep438_model::Entered].load();
  if(!ordinal||(ordinal-1)/layers+1!=forward.load())throw std::logic_error("diagnostic actual forward ordinal differs");
  state_.layer=(ordinal-1)%layers;state_.hidden=clone(r,x);entry=&state_;active_=true;
 }
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
 ~Scope()noexcept{
  if(!active_)return;entry=nullptr;
  // Ordinary success already passed both existing role completion fences.
  // On an exceptional/incomplete path, drain before our owner-stack snapshot
  // can be freed; never defer a GPU Tensor destructor to the CABI reader.
  if(state_.finished!=2){storage[state_.owner].failed=true;try{r_.syncStream();}catch(...){std::terminate();}}
 }
};
struct Shadow{bool active=false;int role=0;rt::Tensor original_gu,gu,clamped,reference_a,reference_p,reference_s,candidate_a,candidate_p,candidate_s,ids,weights;};
template<class Alloc>Shadow prepare(rt::Runtime&r,int role,const rt::Tensor&gu,Alloc alloc,const rt::Tensor&ids={},const rt::Tensor&weights={}){
 Shadow v;if(!entry)return v;r.assertOwnerThread();
 const size_t rows=role==0?6:1;
 if((role!=0&&role!=1)||!deep438::eligible(r,gu)||gu->shape()!=rt::Shape{rows,512})throw std::logic_error("diagnostic actual GU shape");
 v.active=true;v.role=role;v.original_gu=gu;v.gu=alloc({rows,512},rt::DataType::kBFloat16);v.gu->copy_from(r,gu);
 v.clamped=alloc({rows,512},rt::DataType::kBFloat16);v.reference_a=alloc({rows,256},rt::DataType::kBFloat16);
 v.reference_p=alloc({rows,256},rt::DataType::kInt8);v.reference_s=alloc({rows,1},rt::DataType::kFloat32);
 v.candidate_a=alloc({rows,256},rt::DataType::kBFloat16);v.candidate_p=alloc({rows,256},rt::DataType::kInt8);v.candidate_s=alloc({rows,1},rt::DataType::kFloat32);
 if(role==0){
  if(!ids||ids->shape()!=rt::Shape{1,6}||ids->dtype()!=rt::DataType::kInt64||!weights||weights->shape()!=rt::Shape{1,6}||weights->dtype()!=rt::DataType::kFloat32)throw std::logic_error("diagnostic original routes");
  v.ids=alloc({1,6},rt::DataType::kInt64);v.ids->copy_from(r,ids);v.weights=alloc({1,6},rt::DataType::kFloat32);v.weights->copy_from(r,weights);
 }
 ex::submit<infini::ops::Clamp>(r,infini::ops::Clamp::DefaultConfig(r.device().type()),v.gu->narrow({{1,0,256}}),std::optional<double>{},std::optional<double>{10},v.clamped->narrow({{1,0,256}}));
 ex::submit<infini::ops::Clamp>(r,infini::ops::Clamp::DefaultConfig(r.device().type()),v.gu->narrow({{1,256,256}}),std::optional<double>{-10},std::optional<double>{10},v.clamped->narrow({{1,256,256}}));
 ex::submit<infini::ops::SiluAndMul>(r,infini::ops::SiluAndMul::DefaultConfig(r.device().type()),v.clamped,v.reference_a);
 auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(r.device().type());quant.set_implementation_index(0);
 ex::submit<infini::ops::PerChannelQuantI8>(r,ex::RecordingPolicy::HostOnly,quant,v.reference_a,v.reference_p,v.reference_s);
 // Diagnostic candidate output is independent from every main-path operand.
 // The reused original HIP launch writes a BF16 activation only for shadow.
 deep438::apply(r,v.gu,v.candidate_p,v.candidate_s,v.candidate_a);return v;
}
inline TensorBytes copy_completed(rt::Runtime&r,const rt::Tensor&t){
 if(!t||!t->is_contiguous()||t->ndim()!=2)throw std::logic_error("diagnostic tensor layout");
 uint64_t dtype=0,width=0;
 if(t->dtype()==rt::DataType::kBFloat16){dtype=1;width=2;}
 else if(t->dtype()==rt::DataType::kInt8){dtype=2;width=1;}
 else if(t->dtype()==rt::DataType::kFloat32){dtype=3;width=4;}
 else if(t->dtype()==rt::DataType::kInt64){dtype=4;width=8;}
 else throw std::logic_error("diagnostic dtype");
 TensorBytes q;q.info={dtype,t->size(0),t->size(1),width,t->nbytes()};
 if(q.info[4]!=q.info[1]*q.info[2]*width||q.info[4]>8192)throw std::logic_error("diagnostic tensor byte bound");
 q.bytes.resize(t->nbytes());r.memcpyD2H(q.bytes.data(),t->data(),t->nbytes());return q;
}
// Caller invokes this only immediately after its ORIGINAL role r.syncStream().
// That fence completed both the unchanged primary path and our same-stream
// shadow work. Every handle is still in the actual owner Work/owners lease.
inline void completed(rt::Runtime&r,const Shadow&v){
 if(!v.active)return;r.assertOwnerThread();if(!entry||entry->owner!=r.device().index())throw std::logic_error("diagnostic completed owner");
 Record q;q.info={pass.load(),output.load(),forward.load(),entry->layer,uint64_t(entry->owner),uint64_t(v.role),uint64_t(v.role?8:11),0,reinterpret_cast<uintptr_t>(&r),reinterpret_cast<uintptr_t>(r.stream()),std::hash<std::thread::id>{}(std::this_thread::get_id()),reinterpret_cast<uintptr_t>(v.original_gu->data())};
 if(v.role==0){q.fields.push_back(copy_completed(r,entry->hidden));q.fields.push_back(copy_completed(r,v.ids));q.fields.push_back(copy_completed(r,v.weights));}
 for(const auto&t:{v.original_gu,v.gu,v.reference_a,v.reference_p,v.reference_s,v.candidate_a,v.candidate_p,v.candidate_s})q.fields.push_back(copy_completed(r,t));
 for(const auto&t:q.fields)q.info[7]+=t.bytes.size();
 auto&rows=storage[entry->owner].records;if(rows.size()>=86)throw std::length_error("diagnostic max43layers x2roles");rows.push_back(std::move(q));++entry->finished;
}
inline bool any_active(){for(auto&r:deep438_model::rows)if(r.active.load())return true;return false;}
inline int arm(int p,int o,uint64_t f){
 if(p<0||p>=4||(o!=82&&o!=83)||f!=uint64_t(9+o))return 1;std::lock_guard<std::mutex>lock(control);
 if(armed.load()||sealed.load()||any_active())return 2;
 for(size_t i=0;i<8;++i){if(!storage[i].records.empty()||storage[i].failed)return 3;auto&r=deep438_model::rows[i];if(r.value[deep438_model::Entered].load()!=43*(f-1)||r.value[deep438_model::HostReturned].load()!=43*(f-1))return 4;}
 pass.store(p);output.store(o);forward.store(f);armed.store(true);return 0;
}
inline int seal(){std::lock_guard<std::mutex>lock(control);if(!armed.load()||any_active())return 2;size_t bytes=0;
 for(size_t i=0;i<8;++i){if(storage[i].failed||storage[i].records.size()!=86)return 3;std::array<unsigned,86>seen{};
  for(const auto&q:storage[i].records){if(q.info[0]!=pass.load()||q.info[1]!=output.load()||q.info[2]!=forward.load()||q.info[3]>=43||q.info[4]!=i||q.info[5]>1||++seen[q.info[3]*2+q.info[5]]!=1)return 4;bytes+=q.info[7];}
  if(deep438_model::rows[i].value[deep438_model::Entered].load()!=43*forward.load()||deep438_model::rows[i].value[deep438_model::HostReturned].load()!=43*forward.load())return 5;
 }if(bytes>store_limit)return 6;armed.store(false);sealed.store(true);return 0;
}
inline int clear(){std::lock_guard<std::mutex>lock(control);if(armed.load()||any_active())return 2;for(auto&row:storage){row.records.clear();row.failed=false;}sealed.store(false);return 0;}
inline int abandon(){std::lock_guard<std::mutex>lock(control);if(any_active())return 2;armed.store(false);sealed.store(true);return 0;}
inline const Record&record(int owner,size_t index){if(!sealed.load()||owner<0||owner>=8)throw std::invalid_argument("diagnostic sealed CPU record");return storage.at(owner).records.at(index);}
inline int record_info(int owner,size_t index,uint64_t*out,size_t n){if(!out||n!=record_width)return 1;try{auto&q=record(owner,index);std::memcpy(out,q.info.data(),n*sizeof(uint64_t));return 0;}catch(...){return 9;}}
inline int tensor_info(int owner,size_t index,size_t field,uint64_t*out,size_t n){if(!out||n!=tensor_width)return 1;try{auto&q=record(owner,index).fields.at(field);std::memcpy(out,q.info.data(),n*sizeof(uint64_t));return 0;}catch(...){return 9;}}
inline int copy(int owner,size_t index,size_t field,void*out,size_t bytes){if(!out)return 1;try{auto&q=record(owner,index).fields.at(field);if(bytes!=q.bytes.size())return 2;std::memcpy(out,q.bytes.data(),bytes);return 0;}catch(...){return 9;}}
}
