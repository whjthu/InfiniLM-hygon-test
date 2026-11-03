#include "projection.hpp"
// Compile exact real Tensor/Dense/CCL overloads, but never load the object.
// This is syntax/interface qualification only, not an executed GPU consumer.
infinilm::runtime::Tensor compile_r440_actual_calls(
 infinilm::runtime::Runtime&r,const std::shared_ptr<infinilm::nn::Linear>&layer,
 infinilm::runtime::Tensor&input,std::size_t rank,infinicclComm_t comm) {
 return kimi_r440::project(r,layer,input,0,rank,16,comm,true);
}
