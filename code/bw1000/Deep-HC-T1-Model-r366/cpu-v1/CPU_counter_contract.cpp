#include "hc_model_contract_r366.hpp"
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>
int main(){
    assert(deep_hc_model_epilogue_get_r366()==0&&!hc_model_r366::enabled());
    uint64_t out[56]{};assert(deep_hc_model_epilogue_counters_r366(nullptr,56)==1&&deep_hc_model_epilogue_counters_r366(out,55)==1);
    assert(deep_hc_model_epilogue_counters_r366(out,56)==0);for(auto v:out)assert(v==0);
    auto run=[](bool on){std::vector<std::thread>threads;for(int d=0;d<8;++d)threads.emplace_back([=]{for(int n=0;n<86;++n){if(!on){hc_model_r366::on_ctor(d);hc_model_r366::on_ready(d);}hc_model_r366::on_pre(d,1,true);if(hc_model_r366::enabled())hc_model_r366::on_completed(d);if(on)hc_model_r366::on_pre(d,1024,true);}if(on)hc_model_r366::on_pre(d,1,false);});for(auto&t:threads)t.join();};
    run(false);assert(deep_hc_model_epilogue_set_r366(1)==0);run(true);assert(deep_hc_model_epilogue_set_r366(2)==1&&deep_hc_model_epilogue_get_r366()==1);assert(deep_hc_model_epilogue_set_r366(-1)==1);
    assert(deep_hc_model_epilogue_counters_r366(out,56)==0);for(int d=0;d<8;++d){uint64_t expected[]={86,86,259,172,86,86,1};for(int k=0;k<7;++k)assert(out[d*7+k]==expected[k]);}
    assert(deep_hc_model_epilogue_set_r366(0)==0&&!hc_model_r366::enabled());std::cout<<"HC366_CPU_COUNTER_DEFAULT_OFF_8THREAD_PASS\n";
}
