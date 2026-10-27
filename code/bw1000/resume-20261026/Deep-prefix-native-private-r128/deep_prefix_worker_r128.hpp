#pragma once
#include "deep_prefix_transaction_r128.hpp"
#include "models/infinilm_model.hpp"
#include "runtime/runtime.hpp"
namespace infinilm::models::deepseek_v4::prefix_r128 {
enum class Action{Begin,SealPublish,Cancel};
void native_control(::infinilm::InfinilmModel&,runtime::Runtime&,size_t,
 Action,const std::shared_ptr<Group>&,const std::vector<Row>&,const Counts&);
}
