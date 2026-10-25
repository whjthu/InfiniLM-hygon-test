#pragma once
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <thread>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
namespace varlen_metadata_r9141 {
struct Snapshot{size_t requests,tokens,pool;std::vector<int64_t>offsets,reads,writes;};
inline Snapshot validate(size_t n,size_t tokens,size_t pool,const int64_t*off,const int64_t*reads,const int64_t*writes){
 if(!n||n>32||!tokens||tokens>64||(pool!=17&&pool!=33&&pool!=65)||n>(pool-1)/2||!off||!reads||!writes)throw std::invalid_argument("9141 bounded request/T64/pool17,33,65 metadata");Snapshot s{n,tokens,pool,{off,off+n+1},{reads,reads+n},{writes,writes+n}};const int64_t half=int64_t((pool-1)/2);if(s.offsets.front()!=0||s.offsets.back()!=int64_t(tokens))throw std::invalid_argument("9141 offsets endpoint");std::set<int64_t>rows;
 for(size_t i=0;i<n;++i){auto a=s.offsets[i],z=s.offsets[i+1],rr=s.reads[i],ww=s.writes[i];if(a<0||z<=a||z-a>8||z>int64_t(tokens)||rr<1||ww<1||rr>=int64_t(pool)||ww>=int64_t(pool)||std::abs(rr-ww)!=half||std::min(rr,ww)>half||!rows.insert(rr).second||!rows.insert(ww).second)throw std::invalid_argument("9141 positive W<=8 independent paired read/write ownership");}return s;
}
class Ledger{const void*owner_;std::thread::id thread_;uint64_t epoch_=0;public:explicit Ledger(const void*owner):owner_(owner),thread_(std::this_thread::get_id()){if(!owner)throw std::invalid_argument("9141 null owner");}void check(const void*owner,uint64_t epoch)const{if(owner!=owner_||thread_!=std::this_thread::get_id()||!epoch||epoch<=epoch_)throw std::invalid_argument("9141 foreign owner/thread/stale epoch");}void commit(const void*owner,uint64_t epoch){check(owner,epoch);epoch_=epoch;}void own(const void*owner)const{if(owner!=owner_||thread_!=std::this_thread::get_id())throw std::invalid_argument("9141 wrong owner/thread");}uint64_t epoch()const{return epoch_;}};
}
