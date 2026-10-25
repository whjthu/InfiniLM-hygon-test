#pragma once
#include "kimi_pair_state_r9006.hpp"
#include <set>
#include <array>
namespace kimi_batch_logic_r9122 {
struct Request {int logical=0,read=0,write=0,past=0;size_t offset=0,width=0;};
class Transaction {
 const void*owner_;kimi_pair_r9006::State state_;std::vector<size_t>lengths_;uint64_t epoch_=0;std::vector<Request>requests_;
 public:
 Transaction(const void*owner,size_t count,size_t past):owner_(owner),state_(owner,8),lengths_(17,0){if(!count||count>8||!past||past>1024)throw std::invalid_argument("9122 transaction admission");for(size_t j=1;j<=count;++j){state_.prefilled(owner,j,past);lengths_[j]=past;}}
 void preflight(const void*owner,uint64_t epoch,const std::vector<Request>&req){if(owner!=owner_||epoch<=epoch_||req.empty()||req.size()>8)throw std::invalid_argument("9122 owner/epoch/count");std::set<int>ids,physical;size_t offset=0;for(auto x:req){if(!ids.insert(x.logical).second||!physical.insert(x.read).second||!physical.insert(x.write).second||x.offset!=offset||!x.width||x.width>8||size_t(x.read)>=lengths_.size()||x.read<1||size_t(x.past)!=lengths_[x.read])throw std::invalid_argument("9122 per-request offset/owner/CPU ledger");state_.check(owner,x.logical,x.past,x.read,x.write,std::vector<int64_t>(x.width,17));offset+=x.width;}if(offset>16)throw std::invalid_argument("9122 component totalT16 bound");}
 void verified(const void*owner,uint64_t epoch,const std::vector<Request>&req){preflight(owner,epoch,req);auto next=state_;auto lengths=lengths_;for(auto x:req){next.completed(owner,x.logical,std::vector<int64_t>(x.width,17));lengths[x.write]=size_t(x.past)+x.width;}state_=std::move(next);lengths_=std::move(lengths);requests_=req;epoch_=epoch;}
 void commit(const void*owner,uint64_t epoch,const std::vector<size_t>&accepted,const std::vector<int>&continuation){if(owner!=owner_||epoch!=epoch_||accepted.size()!=requests_.size()||continuation.size()!=requests_.size())throw std::invalid_argument("9122 commit epoch/count");auto next=state_;for(size_t i=0;i<requests_.size();++i){if(continuation[i]!=0&&continuation[i]!=1)throw std::invalid_argument("9122 continuation");next.commit(owner,requests_[i].logical,accepted[i],continuation[i]!=0);}state_=std::move(next);}
 std::vector<size_t>retained(const void*owner,uint64_t epoch){if(owner!=owner_||epoch!=epoch_)throw std::invalid_argument("9122 replay epoch");std::vector<size_t>out;for(auto x:requests_){auto&r=state_.row(owner,x.logical);if(r.phase!=1&&r.phase!=3&&r.phase!=4)throw std::invalid_argument("9122 commit required before replay");out.push_back(r.phase==3?r.retained:x.width);}return out;}
 void completed_replay(const void*owner,uint64_t epoch){auto expected=retained(owner,epoch);auto next=state_;auto lengths=lengths_;for(size_t i=0;i<requests_.size();++i){auto x=requests_[i];auto&r=next.row(owner,x.logical);if(r.phase!=3)continue;if(lengths.at(x.read)!=size_t(x.past)||lengths.at(x.write)!=size_t(x.past)+x.width)throw std::invalid_argument("9122 stale verify CPU length ledger");lengths[x.write]=size_t(x.past)+r.retained;next.completed(owner,x.logical,std::vector<int64_t>(r.ids.begin(),r.ids.begin()+r.retained));}state_=std::move(next);lengths_=std::move(lengths);}
 std::array<int32_t,4>row(const void*owner,int logical){auto&r=state_.row(owner,logical);return{r.active,r.shadow,r.past,r.phase};}
 uint64_t epoch()const{return epoch_;}
};
}
