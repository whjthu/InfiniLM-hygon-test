#pragma once

#include "kimi_head_causal_lm_r8341.hpp"
#include "/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_text_model.hpp"
#include "/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_vision.hpp"

#include <memory>

namespace infinilm::models::kimi_k3 {

using KimiK3LanguageModelHeadR8341 = infinilm::layers::causal_lm_templates::TextCausalLMHeadR8341<KimiK3TextModel>;

class KimiK3ForConditionalGenerationHeadR8341 : public infinilm::InfinilmModel {
public:
    KimiK3ForConditionalGenerationHeadR8341(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    Output forward(const Input &input, global_state::ForwardContext &forward_context) const override;
    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank_info) override;

private:
    size_t owner_tp_size_{1}, owner_pp_size_{1}, owner_pp_stage_{0};
    void replace_image_embeddings(infinilm::runtime::Tensor &inputs_embeds,
                                  const Input &input,
                                  global_state::ForwardContext &forward_context) const;

    std::shared_ptr<KimiK3VisionTower> vision_tower_;
    std::shared_ptr<KimiK3Projector> mm_projector_;
    std::shared_ptr<KimiK3LanguageModelHeadR8341> language_model_;
};

std::shared_ptr<infinilm::config::ModelConfig>
create_kimi_k3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::kimi_k3
