#include "varlen_metadata_r9141.hpp"
#include <iostream>
#include <cassert>
#include <future>
using namespace varlen_metadata_r9141;
int main(){size_t checks=0,cases=0;int owner=1,wrong=2;Ledger ledger(&owner);uint64_t epoch=0;
for(size_t capacity:{8ul,16ul,32ul})for(size_t C:{1ul,2ul,4ul,8ul,16ul,32ul})if(C<=capacity)for(size_t W:{1ul,2ul,4ul,8ul})if(C*W<=64)for(int ragged:{0,1}){++cases;std::vector<int64_t>off{0},rr,ww;for(size_t i=0;i<C;++i){off.push_back(off.back()+int64_t(ragged?1+(i%W):W));rr.push_back(int64_t(i+1));ww.push_back(int64_t(i+1+capacity));}auto saved=validate(C,size_t(off.back()),2*capacity+1,off.data(),rr.data(),ww.data());ledger.check(&owner,++epoch);ledger.commit(&owner,epoch);++checks;bool bad=false;try{ledger.check(&owner,epoch);}catch(...){bad=true;}assert(bad);++checks;bad=false;try{ledger.check(&wrong,epoch+1);}catch(...){bad=true;}assert(bad);++checks;
for(int badcase=0;badcase<8;++badcase){auto o=off,a=rr,b=ww;size_t tokens=size_t(off.back()),pool=2*capacity+1;if(badcase==0)o[0]=1;if(badcase==1)o[1]=o[0];if(badcase==2)o[1]=o[0]+9;if(badcase==3)a[0]=0;if(badcase==4)b[0]=a[0];if(badcase==5)pool=18;if(badcase==6)tokens+=1;if(badcase==7)b[0]=int64_t(pool);bad=false;try{validate(C,tokens,pool,o.data(),a.data(),b.data());}catch(...){bad=true;}assert(bad);++checks;}
off[0]=-999;rr[0]=-999;ww[0]=-999;assert(saved.offsets[0]==0&&saved.reads[0]==1&&saved.writes[0]==int64_t(capacity+1));++checks;
if(C>1){auto o=saved.offsets,a=saved.reads,b=saved.writes;a[1]=a[0];b[1]=b[0];bad=false;try{validate(C,size_t(o.back()),2*capacity+1,o.data(),a.data(),b.data());}catch(...){bad=true;}assert(bad);++checks;}
}
assert(std::async(std::launch::async,[&]{try{ledger.check(&owner,epoch+1);return false;}catch(...){return true;}}).get());++checks;std::cout<<"{\"passed\":true,\"pure_CPU\":true,\"cases\":"<<cases<<",\"checks\":"<<checks<<",\"capacity_matrix\":[8,16,32],\"C_matrix\":[1,2,4,8,16,32],\"W_matrix\":[1,2,4,8],\"flattenT_cap\":64,\"GPU_tested\":false}"<<std::endl;}
