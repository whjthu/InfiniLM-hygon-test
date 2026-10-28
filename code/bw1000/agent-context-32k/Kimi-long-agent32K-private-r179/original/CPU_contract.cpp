#include "batch_transaction_r9122.hpp"
#include "batch_payload_r9122c.hpp"
#include <iostream>
#include <cassert>
int main(){size_t payload_checks=0;
 for(size_t C:{8ul,16ul,32ul})for(int64_t op:{1,2,3,4}){size_t count=op<=2?C:0,W=64/C;if(W>8)W=8;std::vector<int64_t>v{91220001,1,op,op<=2?1:0,int64_t(count),int64_t(count*W),int64_t(2*C+1),0};for(size_t i=0;i<count;++i)v.insert(v.end(),{int64_t(i+1),512,int64_t(i+1),int64_t(i+C+1),int64_t(i*W),int64_t(W),op==1?0:1});kimi_batch_payload_r9122c::validate(v.data(),v.size(),2*C+1,op);++payload_checks;
 for(size_t index:{0ul,1ul,6ul,7ul}){auto x=v;x[index]=-1;bool rejected=false;try{kimi_batch_payload_r9122c::validate(x.data(),x.size(),2*C+1,op);}catch(...){rejected=true;}assert(rejected);++payload_checks;}
 if(count){for(size_t index:{8ul,9ul,10ul,11ul,12ul,13ul,14ul}){auto x=v;x[index]=-1;bool rejected=false;try{kimi_batch_payload_r9122c::validate(x.data(),x.size(),2*C+1,op);}catch(...){rejected=true;}assert(rejected);++payload_checks;}}
 }
int owner=1,wrong=2;size_t tests=0;using namespace kimi_batch_logic_r9122;
 for(size_t C:{16ul,32ul}){size_t W=64/C;for(int scenario=0;scenario<3;++scenario){Transaction tx(&owner,C,512);std::vector<Request>req;for(size_t i=0;i<C;++i)req.push_back({int(i+1),int(i+1),int(i+C+1),512,i*W,W});tx.preflight(&owner,1,req);tx.verified(&owner,1,req);std::vector<size_t>a(C);std::vector<int>cont(C,1);for(size_t i=0;i<C;++i)a[i]=(scenario==1?W-1:(i%2?W-1:0));if(scenario==2)cont[0]=0;tx.commit(&owner,1,a,cont);auto kept=tx.retained(&owner,1);tx.completed_replay(&owner,1);std::vector<Request>next;size_t offset=0;for(size_t i=0;i<C;++i){auto r=tx.row(&owner,int(i+1));assert(r[3]==(cont[i]?1:4));if(cont[i]){assert(r[0]==int(i+C+1)&&r[1]==int(i+1)&&r[2]==512+int(a[i]+1));next.push_back({int(i+1),r[0],r[1],r[2],offset++,1});}++tests;}tx.preflight(&owner,2,next);tx.verified(&owner,2,next);++tests;bool rejected=false;try{tx.preflight(&wrong,3,next);}catch(...){rejected=true;}assert(rejected);++tests;rejected=false;try{tx.preflight(&owner,2,next);}catch(...){rejected=true;}assert(rejected);++tests;}}
 std::cout<<"{\"passed\":true,\"pure_CPU\":true,\"checks\":"<<(tests+payload_checks)<<",\"C\":[16,32],\"mixed_retention_EOS_next_forward\":true}"<<std::endl;}
