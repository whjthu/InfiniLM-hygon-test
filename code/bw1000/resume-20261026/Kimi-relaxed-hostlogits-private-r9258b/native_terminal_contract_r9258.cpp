#include "/refactor/incoming/Kimi-speculative-batch-Model-r9122c/kimi_pair_state_r9006.hpp"
#include <cassert>
#include <iostream>
int main(){int owner=0;size_t checks=0;
for(size_t cap:{1u,2u,4u,8u,16u})for(size_t w=1;w<=8;++w)for(size_t accepted=0;accepted<w;++accepted)for(bool continuation:{false,true}){
 kimi_pair_r9006::State s(&owner,cap);s.prefilled(&owner,1,17);
 auto before=s.row(&owner,1);std::vector<int64_t>ids;for(size_t j=0;j<w;++j)ids.push_back(100+j);
 s.completed(&owner,1,ids);s.commit(&owner,1,accepted,continuation);auto after=s.row(&owner,1);
 if(!continuation){assert(after.phase==4&&after.past==before.past&&after.active==before.active&&after.shadow==before.shadow);}
 else if(accepted+1==w){assert(after.phase==1&&after.past==17+int(w)&&after.active==before.shadow);}
 else{assert(after.phase==3&&after.past==17);std::vector<int64_t>prefix(ids.begin(),ids.begin()+accepted+1);s.completed(&owner,1,prefix);auto done=s.row(&owner,1);assert(done.phase==1&&done.past==17+int(accepted)+1&&done.active==before.shadow);}
 ++checks;}
std::cout<<"{\"passed\":true,\"actual_header_metadata_checks\":"<<checks<<",\"GPU\":false,\"native_runtime_created\":false,\"terminal_no_swap_no_past_no_replay\":true}\n";
}