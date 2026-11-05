#include "record_cache_body.hpp"
#include <cassert>
#include <iostream>
int main(){using namespace infinilm::runtime;Runtime r;{auto x=kimi_split_v128::allocate(r);assert(x->shape==Shape({132,256,3,320})&&alive==1&&r.zero_calls==1&&r.sync_calls==0);}assert(alive==0);r.fail=true;try{auto x=kimi_split_v128::allocate(r);assert(false);}catch(const std::runtime_error&){}assert(r.sync_calls==1&&alive==0);std::cout<<"{\"passed\":true,\"production_cache_body_recording\":true,\"failure_drain_while_alive\":true,\"SDK_device_execution\":false}"<<std::endl;}
