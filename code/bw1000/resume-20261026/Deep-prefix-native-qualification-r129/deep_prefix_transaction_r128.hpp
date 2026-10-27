#pragma once
// Private HOST-only opt-in. No SDK/runtime ABI or kernel changes.
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "deep_prefix_group_r128.hpp"
namespace infinilm::models::deepseek_v4::prefix_r128 {
struct Row {size_t row,past,tokens;};
using Counts=std::vector<size_t>; // Same ordering as begin rows; accepted INPUTS1/2.
using Publish=std::function<void()>;
using Prepare=std::function<Publish(const Counts&)>;
struct Epoch {uint64_t value=0;};
class Transaction {
public:
 enum class Phase{Building,ForwardDone,Sealed,Published,Cancelled};
 Transaction(const void*owner,std::shared_ptr<Epoch>epoch,std::vector<Row>rows,uint64_t tx,uint64_t generation)
 :owner_(owner),epoch_(std::move(epoch)),parent_(epoch_->value),thread_(std::this_thread::get_id()),rows_(std::move(rows)),tx_(tx),generation_(generation){
  if(!tx||!generation)throw std::invalid_argument("prefix zero transaction/generation");
  if(!owner_||rows_.empty())throw std::invalid_argument("prefix empty owner/rows");
  for(size_t i=0;i<rows_.size();++i){if(rows_[i].tokens!=2)throw std::invalid_argument("prefix K1 requires exactly2 inputs per row");for(size_t j=0;j<i;++j)if(rows_[i].row==rows_[j].row)throw std::invalid_argument("prefix duplicate row");}
 }
 void check()const{if(thread_!=std::this_thread::get_id()||epoch_->value!=parent_)throw std::invalid_argument("prefix stale or foreign owner thread");}
 void check_owner(const void*p)const{check();if(p!=owner_)throw std::invalid_argument("prefix foreign model");}
 void stage(Prepare p){check();if(phase_!=Phase::Building)throw std::logic_error("prefix stage phase");stages_.push_back(std::move(p));}
 void decoder_done(size_t layer){check();if(layer>=43||layers_[layer])throw std::logic_error("prefix duplicate layer");layers_[layer]=true;}
 void forward_done(){check();for(bool b:layers_)if(!b)throw std::logic_error("prefix missing decoder");if(stages_.size()!=87)throw std::logic_error("prefix needs43attention+43decoder+model stages");phase_=Phase::ForwardDone;}
 void seal(const Counts&c){check();if(phase_!=Phase::ForwardDone||c.size()!=rows_.size())throw std::invalid_argument("prefix seal phase/row count");for(auto n:c)if(n<1||n>2)throw std::invalid_argument("prefix accepted input count must1/2; cancel uses abort");
  std::vector<Publish>ready;ready.reserve(stages_.size());for(auto&p:stages_)ready.push_back(p(c)); // Throw => no publication.
  ready_=std::move(ready);accepted_=c;phase_=Phase::Sealed;
 }
 Stamp stamp()const{check();Stamp s;s.transaction=tx_;s.generation=generation_;s.epoch=parent_;s.accepted=accepted_;for(auto r:rows_)s.cohort.push_back({r.row,r.past,r.tokens});return s;}
 // No local publication is reachable without the real all8 seal permit.
 void publish(const Permit&permit){check();if(phase_!=Phase::Sealed)throw std::logic_error("prefix publish before seal");permit.validate(stamp());for(auto&p:ready_){try{p();}catch(...){std::terminate();}}++epoch_->value;phase_=Phase::Published;ready_.clear();stages_.clear();}
 void cancel(){check();if(phase_==Phase::Published||phase_==Phase::Cancelled)throw std::logic_error("prefix consumed");++epoch_->value;phase_=Phase::Cancelled;ready_.clear();stages_.clear();}
 const std::vector<Row>&rows()const{return rows_;}
 Phase phase()const{return phase_;}
private:
 const void*owner_;std::shared_ptr<Epoch>epoch_;uint64_t parent_;std::thread::id thread_;std::vector<Row>rows_;std::array<bool,43>layers_{};std::vector<Prepare>stages_;std::vector<Publish>ready_;Phase phase_=Phase::Building;uint64_t tx_,generation_;Counts accepted_;
};
inline thread_local Transaction*active=nullptr;
struct Scope{Transaction*previous;explicit Scope(Transaction*t):previous(active){if(previous)throw std::logic_error("prefix nested transaction");active=t;}~Scope(){active=previous;}};
template<class Requests>void validate_rows(const Requests&q){if(!active)return;auto&rows=active->rows();if(q.size()!=rows.size())throw std::invalid_argument("prefix cohort count");for(size_t i=0;i<q.size();++i)if(q[i].row!=rows[i].row||q[i].past!=rows[i].past||q[i].tokens!=rows[i].tokens)throw std::invalid_argument("prefix cohort metadata");}
template<class Requests>void lengths(std::vector<size_t>&target,const Requests&q){
 if(!active){for(auto r:q)target[r.row]=r.past+r.tokens;return;}
 validate_rows(q);auto*dst=&target;active->stage([dst,q](const Counts&c){auto next=std::make_shared<std::vector<size_t>>(*dst);for(size_t i=0;i<q.size();++i)(*next)[q[i].row]=q[i].past+c[i];return [dst,next]()noexcept{dst->swap(*next);};});
}
// Persistent binary trie: updates copy only32 paths, never full token history.
// Slots are bounded by existing native capacity<=2^32; every leaf owns its page.
template<class T>class Slots{
 struct Node;using Ptr=std::shared_ptr<const Node>;
 struct Node{Ptr left,right;std::shared_ptr<const T>value;};
 Ptr root_;
 static Ptr put(Ptr p,uint32_t slot,int bit,std::shared_ptr<const T>value){auto n=std::make_shared<Node>();if(p)*n=*p;if(bit<0)n->value=std::move(value);else if((slot>>bit)&1)n->right=put(n->right,slot,bit-1,std::move(value));else n->left=put(n->left,slot,bit-1,std::move(value));return n;}
public:
 const T*find(uint32_t slot)const{auto p=root_;for(int b=31;b>=0&&p;--b)p=((slot>>b)&1)?p->right:p->left;return p&&p->value?p->value.get():nullptr;}
 Slots with(uint32_t slot,T value)const{Slots n;n.root_=put(root_,slot,31,std::make_shared<T>(std::move(value)));return n;}
 void swap(Slots&other)noexcept{root_.swap(other.root_);}void clear()noexcept{root_.reset();}bool empty()const noexcept{return !root_;}
};
}
