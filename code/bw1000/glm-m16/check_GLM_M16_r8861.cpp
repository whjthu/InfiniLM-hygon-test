#include "glm_model_resident_policy_r5640.hpp"
#include "glm_vector_dispatch_r8540.cpp"
#include <iostream>
static int vec_calls=0,scalar_calls=0;
extern "C" int glm_vec_gate_r8528(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){++vec_calls;return 0;}
extern "C" int glm_vec_down_r8528(void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){++vec_calls;return 0;}
extern "C" int glm_grouped_gate_r5123(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){++scalar_calls;return 0;}
extern "C" int glm_grouped_down_r5123(void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){++scalar_calls;return 0;}
int main(){
 namespace g=glm_model_resident_r5640;g::Slots slots;
 for(size_t base=0;base<32;base+=8){std::vector<g::Request>q;for(size_t i=0;i<8;++i)q.push_back({base+i,0,512});slots=slots.propose(q,32);}
 if(slots.active()!=32)return 1;
 for(size_t base=0;base<32;base+=16){std::vector<g::Request>q;for(size_t i=0;i<16;++i)q.push_back({base+i,512,1});slots=slots.propose(q,32);}
 for(size_t base=0;base<32;base+=8){std::vector<g::Request>q;for(size_t i=0;i<8;++i)q.push_back({base+i,513,1});slots=slots.propose(q,32);}
 for(size_t i=0;i<32;++i)if(slots.length(i)!=514)return 2;
 std::vector<g::Request>oversize;for(size_t i=0;i<17;++i)oversize.push_back({i,514,1});
 bool rejected=false;try{slots.propose(oversize,32);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 3;
 rejected=false;try{slots.propose({{0,514,1},{0,514,1}},32);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 4;
 rejected=false;try{slots.propose({{0,513,1}},32);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 5;
 if(modern_glm_vector_policy_r8540(0,1))return 6;
 {glm_vector_r8540::Frame f(0);for(size_t routes:{32,64,128,256}){glm_vector_r8540::gate(f,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,routes,0,nullptr);glm_vector_r8540::down(f,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,routes,0,nullptr);}}
 if(vec_calls!=6||scalar_calls!=2)return 7;
 if(modern_glm_vector_policy_r8540(0,0))return 8;
 {glm_vector_r8540::Frame f(0);glm_vector_r8540::gate(f,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,128,0,nullptr);}
 if(scalar_calls!=3)return 9;
 std::uint64_t counts[8];if(modern_glm_vector_counts_r8540(0,counts,8)||counts[0]!=3||counts[1]!=2||counts[2]!=3||counts[3]!=1||counts[5]!=28||counts[6]!=28)return 10;
 slots.reset();if(!slots.empty())return 11;
 std::cout<<"{\"actual_Slots_M8_M16\":true,\"M17_duplicate_stale_rejected\":true,\"actual_dispatch_routes32_64_128\":true,\"routes256_fallback\":true,\"OFF_fallback\":true,\"counted_gate_down_tokens\":28,\"GPU\":false,\"dlopen\":false}";
}
