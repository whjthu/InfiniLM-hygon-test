#include "hc_model_contract_r366.hpp"
#include "hc_fullpre_control_r377.hpp"
#include <cassert>
#include <thread>
#include <stdexcept>
#include <vector>
int main(){
 uint64_t out[80]{};assert(deep_hc_model_epilogue_get_r366()==0&&deep_hc_fullpre_enabled_r377()==0);
 assert(deep_hc_model_epilogue_set_r366(2)==1&&deep_hc_model_epilogue_get_r366()==0);
 assert(deep_hc_fullpre_enable_r377(2)==1&&deep_hc_fullpre_enabled_r377()==0);
 assert(deep_hc_fullpre_stats_r377(nullptr,80)==1&&deep_hc_fullpre_stats_r377(out,79)==1);
 {hc_full_r377::Scope off(0,1,true);assert(!off.selected());}
 assert(deep_hc_fullpre_stats_r377(out,80)==0);for(auto v:out)assert(v==0);
 assert(deep_hc_fullpre_enable_r377(1)==0&&deep_hc_model_epilogue_get_r366()==0);assert(deep_hc_fullpre_clear_r377()==1);
 {hc_full_r377::Scope selected(0,1,true);assert(selected.selected());selected.complete();}
 assert(deep_hc_model_epilogue_set_r366(1)==0&&deep_hc_fullpre_enabled_r377()==1);
 {hc_full_r377::Scope prefill(0,297,true);assert(!prefill.selected());}
 {hc_full_r377::Scope f32(0,1,false);assert(!f32.selected());}
 {hc_full_r377::Scope layout(0,1,true,false);assert(!layout.selected());}
 try{hc_full_r377::Scope selected(0,1,true);throw std::runtime_error("selected");}catch(const std::runtime_error&){}
 try{hc_full_r377::Scope fallback(0,1024,false);throw std::runtime_error("fallback");}catch(const std::runtime_error&){}
 assert(deep_hc_fullpre_stats_r377(out,80)==0);uint64_t expected[]={6,3,2,1,2,1,1,1,1,0};for(int i=0;i<10;++i)assert(out[i]==expected[i]);for(int i=10;i<80;++i)assert(out[i]==0);
 assert(deep_hc_fullpre_enable_r377(0)==0&&deep_hc_fullpre_clear_r377()==0);
 assert(deep_hc_fullpre_enable_r377(1)==0);
 {hc_full_r377::Scope active(0,1,true);assert(deep_hc_fullpre_enable_r377(0)==0);assert(deep_hc_fullpre_clear_r377()==1);active.complete();}
 assert(deep_hc_fullpre_clear_r377()==0);assert(deep_hc_fullpre_enable_r377(1)==0);
 std::vector<std::thread> threads;for(int d=0;d<8;++d)threads.emplace_back([d]{for(int i=0;i<32;++i){hc_full_r377::Scope s(d,1,true);assert(s.selected());s.complete();}});for(auto&t:threads)t.join();
 assert(deep_hc_fullpre_stats_r377(out,80)==0);for(int d=0;d<8;++d){uint64_t e[]={32,32,32,32,0,0,0,0,0,0};for(int k=0;k<10;++k)assert(out[d*10+k]==e[k]);}
 assert(deep_hc_model_epilogue_set_r366(2)==1&&deep_hc_model_epilogue_get_r366()==1);
 assert(deep_hc_fullpre_enable_r377(0)==0&&deep_hc_fullpre_clear_r377()==0);
 assert(deep_hc_model_epilogue_get_r366()==1);return 0;
}
