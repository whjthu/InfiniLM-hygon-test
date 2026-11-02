#include "glm_selection_map_r302.hpp"
#include <iostream>
#include <cassert>
int main(){
 using infinilm::models::glm_moe_dsa::selection_map_r302;
 std::vector<int64_t>d;for(int i=0;i<2047;++i)d.push_back(i*16);d.push_back(32767);
 auto dm=selection_map_r302(d,1,2048,32767);assert(dm.logical.size()==2048);
 std::vector<int64_t>p;for(int t=0;t<512;++t){for(int j=0;j<2047;++j)p.push_back((t*64+j)%32256);p.push_back(32256+t);}
 auto pm=selection_map_r302(p,512,2048,32256);assert(pm.logical.size()==32768);
 for(size_t i=0;i<32768;++i)assert(pm.logical[i]==int64_t(i));
 bool bad=false;d[0]=32768;try{selection_map_r302(d,1,2048,32767);}catch(...){bad=true;}assert(bad);
 bad=false;try{selection_map_r302({},1,2048,32767);}catch(...){bad=true;}assert(bad);
 std::cout<<"{\"passed\":true,\"actual_parent_selection_map\":true,\"decode_U\":2048,\"prefill512_U\":32768,\"invalid_boundary_rejected\":true,\"GPU\":0}\n";
}
