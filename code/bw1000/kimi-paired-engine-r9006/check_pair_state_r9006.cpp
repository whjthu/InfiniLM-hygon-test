#include "kimi_pair_state_r9006.hpp"
#include <iostream>
using namespace kimi_pair_r9006;
int main(){int owner=1,other=2;size_t positive=0,negative=0;
auto reject=[&](auto f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}if(!caught)throw std::runtime_error("invalid accepted");++negative;};
for(size_t B:{1ul,8ul,16ul,64ul}){Geometry g(B);if(g.state_rows()!=std::max(16ul,2*B)+1||g.pages()!=4*g.state_rows())return 1;
for(int past:{1,252,255,256,1016})for(size_t m=0;m<8;++m){State s(&owner,B);s.prefilled(&owner,1,past);auto initial=s.row(&owner,1);std::vector<int64_t>ids{10,11,12,13,14,15,16,17};s.check(&owner,1,past,initial.active,initial.shadow,ids);s.completed(&owner,1,ids);s.commit(&owner,1,m,true);
if(m<7){auto a=s.row(&owner,1);if(a.past!=past||a.active!=initial.active||a.phase!=3)return 2;ids.resize(m+1);s.check(&owner,1,past,a.active,a.shadow,ids);s.completed(&owner,1,ids);}
auto a=s.row(&owner,1);if(a.past!=past+int(m+1)||a.active!=initial.shadow||a.phase!=1)return 3;
// Canonical KV row remains1 even when recurrent active is half+1.
size_t firstpage=1;size_t slot=(firstpage+size_t(past)/256)*256+size_t(past)%256;if(slot>=g.pages()*256)return 4;++positive;
State terminal(&owner,B);terminal.prefilled(&owner,1,past);terminal.completed(&owner,1,{10,11});terminal.commit(&owner,1,0,false);reject([&]{terminal.check(&owner,1,past,1,int(g.half+1),{10});});
}}
State s(&owner,8);reject([&]{s.row(&other,1);});reject([&]{s.row(&owner,9);});reject([&]{s.check(&owner,1,0,1,9,{1});});s.prefilled(&owner,1,255);reject([&]{s.prefilled(&owner,1,255);});reject([&]{s.check(&owner,1,256,1,9,{1});});reject([&]{s.check(&owner,1,255,9,1,{1});});reject([&]{s.check(&owner,1,255,1,1,{1});});s.completed(&owner,1,{1,2,3,4,5,6,7,8});reject([&]{s.commit(&owner,1,8,true);});s.commit(&owner,1,2,true);reject([&]{s.check(&owner,1,255,1,9,{1,2});});reject([&]{s.check(&owner,1,255,1,9,{1,2,99});});s.poison();reject([&]{s.row(&owner,1);});
Geometry a(8),b(16);if(a.extra_vs_ordinary16_per_layer()!=0||b.extra_vs_ordinary16_per_layer()!=3366912)return 5;
std::cout<<"{\"passed\":true,\"positive\":"<<positive<<",\"negative\":"<<negative<<",\"B8_pool\":17,\"B16_pool\":33,\"B16_extra_recurrent_bytes_per_local_KDAlayer\":3366912,\"GPU\":false}"<<std::endl;
}
