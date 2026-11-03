#include "pending_leases.hpp"
#include "single_row_span.hpp"
#include <memory>
#include <functional>
#include <string>
#include <stdexcept>
#include <iostream>
using namespace kimi_r440;
int main(){int cases=0,drains=0;bool alive_at_completion=false;std::weak_ptr<int>w;{
 auto t=std::make_shared<int>(1);w=t;PendingLeases<std::shared_ptr<int>,std::function<void()>> s([&]{++drains;alive_at_completion=!w.expired();},true);s.retain(t);t.reset();if(w.expired())throw std::runtime_error("early release");s.complete();if(!w.expired()||!alive_at_completion||drains!=1)throw std::runtime_error("success completion order");++cases;bool reject=false;try{s.retain({});}catch(const std::logic_error&){reject=true;}if(!reject)throw std::runtime_error("closed retain");++cases;}
 if(drains!=1)throw std::runtime_error("duplicate drain");++cases;
 {auto t=std::make_shared<int>(2);w=t;try{PendingLeases<std::shared_ptr<int>,std::function<void()>>s([&]{++drains;if(w.expired())throw std::runtime_error("exception early release");},true);s.retain(t);t.reset();throw std::runtime_error("primary");}catch(const std::runtime_error&e){if(std::string(e.what())!="primary")throw;}if(!w.expired()||drains!=2)throw std::runtime_error("exception completion");++cases;}
 {PendingLeases<std::shared_ptr<int>,std::function<void()>>s([&]{++drains;},false);s.complete();if(drains!=3)throw std::runtime_error("symmetric baseline drain");++cases;}
 for(auto k:{7168u,3584u}){for(auto pitch:{1LL,(long long)k,2LL*k}){if(!physical_single_row(2,1,k,k,pitch,1))throw std::runtime_error("legal singleton");++cases;}for(auto pitch:{0LL,-1LL}){if(physical_single_row(2,1,k,k,pitch,1))throw std::runtime_error("illegal pitch");++cases;}if(physical_single_row(3,1,k,k,k,1)||physical_single_row(2,2,k,k,k,1)||physical_single_row(2,1,k,k,k,2))throw std::runtime_error("layout broadening");cases+=3;if(!inside_owned_span(4096,k,4096,4*k)||inside_owned_span(4096+2*k+1,k,4096,4*k)||inside_owned_span(4095,k,4096,4*k))throw std::runtime_error("address envelope");cases+=3;}
 std::cout<<"{\"passed\":true,\"changed_contract_cases\":"<<cases<<",\"recording_completion_calls\":"<<drains<<",\"SDK_execution\":false,\"old35geometry_replayed\":false}"<<std::endl;
}
