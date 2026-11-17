#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace infinilm::models::glm_moe_dsa::v_on_demand_r403 {
inline bool requested(){
 const char*p=std::getenv("GLM_V_ON_DEMAND_R403");
 if(p&&std::string(p)!="0"&&std::string(p)!="1")throw std::invalid_argument("r403 explicit 0/1 policy required");
 return p&&std::string(p)=="1";
}
}
