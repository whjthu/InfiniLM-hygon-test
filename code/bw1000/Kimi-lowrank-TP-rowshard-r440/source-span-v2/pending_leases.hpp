#pragma once
#include <vector>
#include <exception>
#include <stdexcept>
#include <utility>
namespace kimi_r440 {
// Exact per-full-forward lease set, not a cache. Drain is the caller's existing
// completion operation; refs are never cleared before that callback succeeds.
template<class T,class Drain> class PendingLeases {
 std::vector<T>refs_;Drain drain_;bool closed_=false;
public:
 PendingLeases(Drain drain,bool enabled):drain_(std::move(drain)){if(enabled)refs_.reserve(14);}
 PendingLeases(const PendingLeases&)=delete;PendingLeases&operator=(const PendingLeases&)=delete;
 void retain(const T&t){if(closed_||refs_.size()>=14)throw std::logic_error("440 inactive/full pending lease scope");refs_.push_back(t);}
 size_t size()const{return refs_.size();}bool open()const{return !closed_;}
 void complete(){if(closed_)throw std::logic_error("440 duplicate completion");drain_();refs_.clear();std::vector<T>().swap(refs_);closed_=true;}
 ~PendingLeases()noexcept{if(!closed_)try{complete();}catch(...){std::terminate();}}
};
}
