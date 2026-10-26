#include "lease32_policy_r9205.hpp"
#include <cassert>
#include <iostream>
int main(){using glm_lease32_r9205::use;assert(use(1,0,16,16,0));assert(!use(1,0,32,32,0));assert(use(1,1,17,17,0));assert(use(1,1,32,32,0));assert(!use(1,1,33,33,0));assert(!use(1,1,64,32,0));assert(!use(1,1,32,32,1));assert(!use(0,1,32,32,0));std::cout<<"{\"CPU_only\":true,\"GPU\":false,\"actual_gate_boundary_cases\":8,\"defaultOFF_preserves_le16\":true}";}
