#pragma once
#include "models/kimi_k3/kimi_k3_moe.hpp"
namespace infinilm::models::kimi_k3 {
class KimiK3MoER440 : public infinilm::weights::ModelWeights {
public:
    KimiK3MoER440(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> routed_expert_down_proj_;
    std::shared_ptr<KimiK3Experts> experts_;
    std::shared_ptr<infinilm::layers::RMSNorm> routed_expert_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> routed_expert_up_proj_;
    std::shared_ptr<KimiK3MLP> shared_experts_;
    mutable infinilm::runtime::Tensor tiled_scratch_r1861_;
    size_t stable_down_layer_r4012_{0};
    int tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};
}
