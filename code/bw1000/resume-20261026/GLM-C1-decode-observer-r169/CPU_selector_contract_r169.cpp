#include "glm_decode_selector_r169.hpp"
#include <cassert>
#include <iostream>
int main(){
 std::uint64_t seen=0;std::size_t p[]={0,0,100},d[]={0,100,1};
 assert(!glm_decode_r169::capture(100,1,p,seen)&&seen==0);
 assert(!glm_decode_r169::capture(1,1,nullptr,seen)&&seen==0);
 assert(!glm_decode_r169::capture(16,16,d,seen)&&seen==0);
 assert(!glm_decode_r169::capture(15,15,d,seen)&&seen==0);
 int hits=0;for(int i=1;i<=3000;++i){bool got=glm_decode_r169::capture(1,1,d,seen);assert(got==(i>=17&&i<=19));hits+=got;}
 assert(hits==3&&seen==3000);seen=0;for(int i=1;i<=19;++i)hits+=glm_decode_r169::capture(1,1,d,seen);assert(hits==6);
 std::cout<<"{\"passed\":true,\"CPU_only\":true,\"GPU\":false,\"model\":false,\"selection_window\":[17,18,19],\"prefix_and_T15_T16_rejected\":true,\"bounded_samples\":3}";
}
