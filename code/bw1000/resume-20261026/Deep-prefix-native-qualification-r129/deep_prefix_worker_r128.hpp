#pragma once
#include "deep_prefix_transaction_r128.hpp"
#include "models/infinilm_model.hpp"
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
namespace infinilm::models::deepseek_v4::prefix_r128 {
nlohmann::json native_inspect_r129(::infinilm::InfinilmModel&,runtime::Runtime&,size_t,const std::vector<size_t>&);
enum class Action{Begin,SealPublish,Cancel};
void native_control(::infinilm::InfinilmModel&,runtime::Runtime&,size_t,
 Action,const std::shared_ptr<Group>&,const std::vector<Row>&,const Counts&);
}
