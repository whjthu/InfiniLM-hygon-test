#include "glm_resident_exact_budget_r8854.hpp"
#include <iostream>
int main(){
 using namespace glm_resident_exact_r8854;namespace g=glm_model_resident_r5640;
 static_assert(rows32_bank_bytes==167772160);
 const auto b=admission_bytes(32,40);if(b!=6710886400ull)return 1;
 g::admit(b+g::reserve_bytes+g::new_workspace_allowance,b);
 bool rejected=false;try{g::admit(b+g::reserve_bytes+g::new_workspace_allowance-1,b);}catch(const std::invalid_argument&){rejected=true;}
 if(!rejected||admission_bytes(16,1)!=128ull*1024*1024||admission_bytes(64,1)!=g::bank_bytes)return 2;
 std::cout<<"{\"resident_bank_bytes\":"<<rows32_bank_bytes<<",\"layers40_bytes\":"<<b<<",\"reserve_bytes\":"<<g::reserve_bytes<<",\"allowance_bytes\":"<<g::new_workspace_allowance<<",\"boundary_success\":true,\"onebyte_short_rejected\":true,\"GPU\":false}";
}
