#include "deep_smallt_policy_r9228.hpp"
#include <stdexcept>
#include <iostream>
int main(){int count=0;for(auto n:{0ul,1ul,2ul,4ul,8ul,16ul,32ul,33ul,64ul,2048ul}){bool expected=n>=2&&n<=32;if(deep_smallt_r9228::allow(n,1,1)!=expected)throw std::logic_error("scope boundary");if(deep_smallt_r9228::allow(n,1,0)||deep_smallt_r9228::allow(n,0,1)||deep_smallt_r9228::allow(n,2,1))throw std::logic_error("default/mode boundary");++count;}std::cout<<"{\"passed\":true,\"actual_scope_cases\":"<<count<<",\"B1_original\":true,\"mode2_original\":true,\"PF33plus_original\":true}";}
