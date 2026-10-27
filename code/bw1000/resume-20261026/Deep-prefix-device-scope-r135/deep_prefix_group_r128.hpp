#pragma once
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace infinilm::models::deepseek_v4::prefix_r128 {
struct Stamp {
 uint64_t transaction=0,generation=0,epoch=0;
 std::vector<std::array<size_t,3>> cohort;
 std::vector<size_t> accepted;
 bool operator==(const Stamp&o)const{return transaction==o.transaction&&generation==o.generation&&epoch==o.epoch&&cohort==o.cohort&&accepted==o.accepted;}
};
class Group;
class Permit {
 friend class Group;
 const Group*group_;size_t rank_;Stamp stamp_;
 Permit(const Group*g,size_t r,const Stamp&s):group_(g),rank_(r),stamp_(s){}
public:
 void validate(const Stamp&s)const;
 size_t rank()const{return rank_;}
};
// One process, exactly8 Runtime owner threads. No GPU/communication in this
// host agreement. Callers MUST drain their owner before arrive_sealed().
class Group {
 mutable std::mutex mutex_;std::condition_variable cv_;
 std::array<bool,8>sealed_{},published_{},cancelled_{};
 std::array<std::array<uint64_t,5>,8>free_bytes_{};int fault_rank_=-1,fault_phase_=0;
 Stamp common_;bool have_stamp_=false,poisoned_=false,cancelling_=false;
 std::string reason_;uint64_t tx_,generation_;size_t sealed_count_=0,published_count_=0;
 void check_locked()const{if(poisoned_)throw std::runtime_error("prefix cohort poisoned: "+reason_);if(cancelling_)throw std::runtime_error("prefix cohort cancelled");}
public:
 explicit Group(uint64_t tx,uint64_t generation):tx_(tx),generation_(generation){if(!tx||!generation)throw std::invalid_argument("prefix zero transaction/generation");}
 uint64_t transaction()const{return tx_;}uint64_t generation()const{return generation_;}
 void poison(std::string reason)noexcept{try{std::lock_guard<std::mutex>lock(mutex_);poisoned_=true;reason_=std::move(reason);cv_.notify_all();}catch(...){std::terminate();}}
 Permit arrive_sealed(size_t rank,const Stamp&s){
  std::unique_lock<std::mutex>lock(mutex_);check_locked();
  if(rank>=8||sealed_[rank]||s.transaction!=tx_||s.generation!=generation_||s.cohort.empty()||s.cohort.size()!=s.accepted.size()){
   poisoned_=true;reason_="invalid or duplicate seal";cv_.notify_all();throw std::invalid_argument(reason_);
  }
  for(size_t i=0;i<s.cohort.size();++i)if(s.cohort[i][2]!=2||s.accepted[i]<1||s.accepted[i]>2){poisoned_=true;reason_="invalid accepted vector";cv_.notify_all();throw std::invalid_argument(reason_);}
  if(!have_stamp_){common_=s;have_stamp_=true;}else if(!(s==common_)){poisoned_=true;reason_="transaction/epoch/cohort/accepted disagreement";cv_.notify_all();throw std::invalid_argument(reason_);}
  sealed_[rank]=true;++sealed_count_;cv_.notify_all();
  if(!cv_.wait_for(lock,std::chrono::seconds(60),[&]{return poisoned_||cancelling_||sealed_count_==8;})){poisoned_=true;reason_="all8 seal wait timeout";cv_.notify_all();}
  check_locked();return Permit(this,rank,s);
 }
 void validate(const Permit&p,const Stamp&s)const{
  std::lock_guard<std::mutex>lock(mutex_);check_locked();
  if(p.group_!=this||p.rank_>=8||sealed_count_!=8||!sealed_[p.rank_]||published_[p.rank_]||!(s==common_)||!(p.stamp_==s))throw std::logic_error("prefix invalid publication permit");
 }
 void published(const Permit&p){
  std::unique_lock<std::mutex>lock(mutex_);check_locked();
  if(p.group_!=this||p.rank_>=8||sealed_count_!=8||published_[p.rank_])throw std::logic_error("prefix duplicate publication");
  published_[p.rank_]=true;++published_count_;cv_.notify_all();
  if(!cv_.wait_for(lock,std::chrono::seconds(60),[&]{return poisoned_||published_count_==8;})){poisoned_=true;reason_="all8 publication wait timeout";cv_.notify_all();}
  check_locked();
 }
 void start_cancel(){std::lock_guard<std::mutex>lock(mutex_);if(published_count_)throw std::logic_error("prefix cancel after any publication");cancelling_=true;cv_.notify_all();}
 void cancelled(size_t rank){std::lock_guard<std::mutex>lock(mutex_);if(rank>=8||cancelled_[rank])throw std::logic_error("prefix duplicate cancellation");cancelled_[rank]=true;}
 void configure_fault_r129(int rank,int phase){std::lock_guard<std::mutex>lock(mutex_);if(rank<0||rank>=8||phase<0||phase>2||sealed_count_)throw std::invalid_argument("prefix qualification fault config");fault_rank_=rank;fault_phase_=phase;}
 bool fault_before_seal_r129(size_t rank)const{std::lock_guard<std::mutex>lock(mutex_);return fault_phase_==1&&int(rank)==fault_rank_;}
 void before_publish_r129(size_t rank){std::unique_lock<std::mutex>lock(mutex_);check_locked();if(fault_phase_==2&&int(rank)==fault_rank_){if(!cv_.wait_for(lock,std::chrono::seconds(60),[&]{return poisoned_||published_count_>0;})){poisoned_=true;reason_="qualification partial-publish wait timeout";cv_.notify_all();}check_locked();poisoned_=true;reason_="injected after another actual owner publication";cv_.notify_all();throw std::runtime_error(reason_);}}
 void memory_r129(size_t rank,size_t phase,uint64_t bytes){std::lock_guard<std::mutex>lock(mutex_);if(rank>=8||phase>=5)throw std::invalid_argument("prefix memory metadata");free_bytes_[rank][phase]=bytes;}
 std::array<std::array<uint64_t,5>,8>memory_r129()const{std::lock_guard<std::mutex>lock(mutex_);return free_bytes_;}
 bool complete()const{std::lock_guard<std::mutex>lock(mutex_);return !poisoned_&&published_count_==8;}
 bool fully_cancelled()const{std::lock_guard<std::mutex>lock(mutex_);for(bool c:cancelled_)if(!c)return false;return published_count_==0;}
 bool poisoned()const{std::lock_guard<std::mutex>lock(mutex_);return poisoned_;}
 std::array<size_t,3>counts()const{std::lock_guard<std::mutex>lock(mutex_);size_t cancel=0;for(bool c:cancelled_)cancel+=c;return {sealed_count_,published_count_,cancel};}
};
inline void Permit::validate(const Stamp&s)const{if(!group_)throw std::logic_error("prefix missing permit");group_->validate(*this,s);}
}
