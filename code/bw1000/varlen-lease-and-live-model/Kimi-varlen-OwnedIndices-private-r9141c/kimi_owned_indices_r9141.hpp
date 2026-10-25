#pragma once
#include "runtime/tensor.hpp"
#include <array>
#include <set>
#include <cstdint>
#include <memory>
#include <vector>
namespace infinilm::models::kimi_native {
namespace rt=infinilm::runtime;
// Internal eager decode scope. No default SDK/provider or graph eligibility change.
class OwnedIndicesRequestR9141 final {
public:
 OwnedIndicesRequestR9141(rt::Runtime&runtime,const rt::Tensor&offsets,const rt::Tensor&reads,const rt::Tensor&writes,const std::vector<rt::Tensor>&conv_pools,const std::vector<rt::Tensor>&ssm_pools,uint64_t epoch,const void*owner);
 ~OwnedIndicesRequestR9141()noexcept;
 OwnedIndicesRequestR9141(const OwnedIndicesRequestR9141&)=delete;
 OwnedIndicesRequestR9141&operator=(const OwnedIndicesRequestR9141&)=delete;
 void finish();
 bool matches(rt::Runtime&runtime)const;
 void kda(const std::array<rt::Tensor,9>&values,float scale,float lower,bool normalize);
private:
 std::set<const void*>state_owners_;
 rt::Runtime*runtime_;std::shared_ptr<rt::Memory>metadata_,workspace_;void*raw_{};
};
void owned_indices_clear_r9141(rt::Runtime&,const void*owner);
bool owned_indices_active_r9141(rt::Runtime&runtime);
void owned_indices_kda_r9141(rt::Runtime&runtime,const rt::Tensor&out,const rt::Tensor&state,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&gate,const rt::Tensor&beta,const rt::Tensor&alog,const rt::Tensor&bias,float scale,float lower,bool normalize);
}
