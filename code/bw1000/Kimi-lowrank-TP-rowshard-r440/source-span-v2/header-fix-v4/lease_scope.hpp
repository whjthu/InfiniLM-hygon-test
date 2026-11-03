#pragma once
#include "pending_leases.hpp"
#include <infiniccl/infiniccl.h>
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <array>
#include <functional>
#include <stdexcept>
namespace kimi_r440 {
namespace rt=infinilm::runtime;
class ProjectionLeaseScope;
inline thread_local ProjectionLeaseScope*active_scope=nullptr;
class ProjectionLeaseScope {
 rt::Runtime&r_;infinicclComm_t comm_;int rank_,tp_,node_,local_;bool enabled_;
 using LeaseSet=PendingLeases<rt::Tensor,std::function<void()>>;LeaseSet leases_;
 std::array<unsigned,2>counts_{};size_t retained_before_complete_=0;
 static rt::Runtime&checked(rt::Runtime&r,int rank,int tp,infinicclComm_t comm,int node,int local){r.assertOwnerThread();if(active_scope||tp!=16||rank<0||rank>=16||!comm||node<0||node>1||local<0||local>=8||rank!=node*8+local||r.device().index()!=local)throw std::invalid_argument("440 actual owner/TP16/comm scope binding");return r;}
public:
 ProjectionLeaseScope(rt::Runtime&r,int rank,int tp,infinicclComm_t comm,bool enabled,int node,int local):r_(checked(r,rank,tp,comm,node,local)),comm_(comm),rank_(rank),tp_(tp),node_(node),local_(local),enabled_(enabled),leases_([&r]{r.syncStream();},enabled){
  active_scope=this;
 }
 ProjectionLeaseScope(const ProjectionLeaseScope&)=delete;
 ~ProjectionLeaseScope()noexcept{if(leases_.open())try{leases_.complete();}catch(...){std::terminate();}active_scope=nullptr;}
 bool enabled()const{return enabled_;}int rank()const{return rank_;}int tp()const{return tp_;}
 void require(rt::Runtime&r,size_t tp,infinicclComm_t comm)const{r.assertOwnerThread();if(active_scope!=this||!enabled_||!leases_.open()||&r!=&r_||tp!=size_t(tp_)||comm!=comm_)throw std::invalid_argument("440 enqueue requires live exact owning scope");}
 void retain(const rt::Tensor&t){leases_.retain(t);}
 void submitted(int which){if(which<0||which>1||counts_[which])throw std::logic_error("440 duplicate projection within C1 forward");++counts_[which];}
 const std::array<unsigned,2>&counts()const{return counts_;}
 size_t retained_before_complete()const{return retained_before_complete_;}
 void complete_original_forward(){r_.assertOwnerThread();retained_before_complete_=leases_.size();leases_.complete();}
};
}
