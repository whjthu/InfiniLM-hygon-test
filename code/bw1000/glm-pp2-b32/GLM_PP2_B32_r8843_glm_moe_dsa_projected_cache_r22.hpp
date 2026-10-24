#pragma once
#include "glm_moe_dsa_native_attention.hpp"
#include "glm_projected_host_r4504.hpp"
#include "glm_model_resident_policy_r5640.hpp"
namespace infinilm::models::glm_moe_dsa {
class NativeAttentionProjectedR22 final:public weights::ModelWeights {
public:
 using Request=IndexerSelectionR1581::Request;
 struct Output{runtime::Tensor hidden;std::vector<runtime::Tensor>selection;AttentionPrepare::Output prepared;runtime::Tensor attention;PrefixProofsR1645 selection_proofs;};
 NativeAttentionProjectedR22(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows,size_t capacity,size_t topk,const bool*bounded_policy=nullptr,const bool*owned_prefix_policy=nullptr);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&hidden,const runtime::Tensor&positions,const std::vector<Request>&,const std::vector<runtime::Tensor>&shared_selection={},const PrefixProofsR1645&shared_proofs={});
 void reset(runtime::Runtime&);
 using ResidentBuffers=std::pair<runtime::Tensor,runtime::Tensor>;
 ResidentBuffers resident_prepare_r5640(runtime::Runtime&,bool);
 void resident_publish_r5640(const bool*,ResidentBuffers)noexcept;
 void resident_validate_r5640(runtime::Runtime&,const std::vector<Request>&)const;
 nlohmann::json resident_report_r5640()const;

 const runtime::Tensor&latent_cache()const{return latent_;}
 const runtime::Tensor&rotary_cache()const{return rotary_;}
 runtime::Tensor index_cache()const{return selector_?selector_->cache():runtime::Tensor{};}
 const runtime::Tensor&projected_key_cache()const{return projected_key_;}
 const runtime::Tensor&projected_value_cache()const{return projected_value_;}
private:
 // Borrowed from the stable owning CAPI Context; changed only after owner drain/reset.
 runtime::Runtime*owner_;const bool*bounded_policy_r1406_=nullptr;const bool*owned_prefix_policy_r1669_=nullptr;size_t rows_,capacity_,maximum_,topk_;bool full_,ready_=false,poisoned_=false;
 std::shared_ptr<AttentionPrepare>prepare_;std::shared_ptr<AttentionOutput>output_;
 std::shared_ptr<NativeIndexer>indexer_;std::unique_ptr<IndexerSelectionR1581>selector_;
 runtime::Tensor latent_,rotary_,projected_key_,projected_value_;std::vector<size_t>lengths_;
 std::unique_ptr<projected_host_r4504::Bank>host_bank_r4504_;
 // Policy address belongs to CAPI Context; module is destroyed before Context.
 const bool*resident_policy_r5640_=nullptr;
 runtime::Tensor resident_key_r5640_,resident_value_r5640_;
 glm_model_resident_r5640::Slots resident_slots_r5640_;
 uint64_t resident_append_bytes_r5640_=0,resident_shared_calls_r5640_=0,resident_row601_calls_r5640_=0;

};
}
