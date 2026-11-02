#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace infinilm::models::glm_moe_dsa::kv_on_demand_r408 {
inline bool requested(bool noV){
 const char*p=std::getenv("GLM_SELECTED_KV_ON_DEMAND_R408");
 if(p&&std::string(p)!="0"&&std::string(p)!="1")throw std::invalid_argument("r408 explicit 0/1 policy required");
 bool on=p&&std::string(p)=="1";if(on&&!noV)throw std::invalid_argument("r408 requires original r403 NoV");return on;
}
inline void scope(bool on,bool noV,size_t rows,size_t capacity,size_t topk,int tp,bool host){if(on&&(!noV||rows!=1||capacity!=32768||topk!=2048||tp!=8||host))throw std::invalid_argument("r408 requires NoV C1/full32768/topK2048/TP8/device cache");}
}
