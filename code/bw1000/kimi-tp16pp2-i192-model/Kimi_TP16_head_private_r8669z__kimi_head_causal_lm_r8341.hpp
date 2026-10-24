#pragma once
#include "/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/layers/causal_lm_templates/text_causal_lm.hpp"
#include "kimi_dual_vocab_head_r8341.hpp"
namespace infinilm::layers::causal_lm_templates {
template <typename Model>
class TextCausalLMHeadR8341 : public InfinilmModel {
public:
    TextCausalLMHeadR8341(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
                 const engine::distributed::RankInfo &rank_info, backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep_config) {
        models::validate_runtime_construction(model_config, runtime, rank_info);
        const auto head_output_dtype = detail::prepare_lm_head_output_dtype(
            *model_config, runtime.device(), model_config->get_dtype());
        int32_t shift_value = -1;
        infinilm::runtime::Tensor shift_host; // Live until BODY catch drain or successful fence.
        try {

            model_config_ = model_config;

            size_t hidden_size = model_config->get<size_t>("hidden_size");
            size_t vocab_size = model_config->get<size_t>("vocab_size");
            const auto &dtype{model_config->get_dtype()};
            pp_size_ = static_cast<size_t>(rank_info.pp_size);
            pp_stage_ = static_cast<size_t>(rank_info.pp_stage);

            model_ = this->bind_child("model", std::make_shared<Model>(runtime, model_config, rank_info, attention_backend, ep_config));
            if (is_last_pp_stage()) {
                lm_head_ = this->bind_child("lm_head", std::make_shared<infinilm::layers::linear::DualVocabHeadR8341>(runtime, hidden_size, vocab_size, false, dtype, head_output_dtype, rank_info));
                shift_host = infinilm::runtime::Tensor::from_blob(&shift_value, {1}, infinilm::runtime::DataType::kInt32,
                                                                  infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
                last_token_shift_ = infinilm::runtime::Tensor::empty(runtime, {1}, infinilm::runtime::DataType::kInt32);
                last_token_shift_->copy_from(runtime, shift_host);
            }
            if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
                runtime.syncStream();
            }
        } catch (...) {
            if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
                try {
                    runtime.syncStream();
                } catch (...) { std::terminate(); }
            }
            throw;
        }
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input,
                   infinilm::global_state::ForwardContext &forward_context) const override {
        if (forward_context.runtime == nullptr) {
            throw std::invalid_argument("language model forward requires an explicit Runtime");
        }
        auto hidden_states = model_->forward(input, forward_context);
        if (!is_last_pp_stage()) {
            return {infinilm::runtime::Tensor(), hidden_states};
        }

        auto lm_head_input = hidden_states;
        if (!input.sample_all_positions && input.input_offsets.has_value()) {
            const size_t num_requests = input.input_offsets.value()->numel() - 1;
            const bool is_packed_prefill = hidden_states->ndim() == 3
                                        && hidden_states->size(0) == 1
                                        && hidden_states->size(1) > num_requests;
            if (is_packed_prefill) {
                auto end_offsets = input.input_offsets.value()->narrow(
                    {{0, 1, num_requests}});
                auto last_token_positions = infinilm::runtime::Tensor::empty((*forward_context.runtime), end_offsets->shape(), end_offsets->dtype());
                infinilm::execution::submit<infini::ops::Add>(
                    (*forward_context.runtime), infini::ops::Add::DefaultConfig((*forward_context.runtime).device().type()), end_offsets, last_token_shift_, last_token_positions);
                INFINILM_TRACE_OP(Add, last_token_positions);
                auto packed_hidden = hidden_states->view(
                    {hidden_states->size(1), hidden_states->size(2)});
                auto gathered_hidden = infinilm::runtime::Tensor::empty(
                    *forward_context.runtime, {num_requests, hidden_states->size(2)}, packed_hidden->dtype());
                infinilm::execution::submit<infini::ops::Embedding>(
                    *forward_context.runtime,
                    infini::ops::Embedding::DefaultConfig(forward_context.runtime->device().type()),
                    last_token_positions, packed_hidden, gathered_hidden);
                INFINILM_TRACE_OP(Embedding, gathered_hidden);
                lm_head_input = gathered_hidden->view({1, num_requests, hidden_states->size(2)});
            }
        }

        auto logits = lm_head_->forward(*forward_context.runtime, lm_head_input);
        return {logits, hidden_states};
    }

    bool supports_kv_only_prepare() const override {
        return detail::KvOnlyPrepareCapability<Model>::value;
    }

    infinilm::runtime::Tensor logits_from_hidden(infinilm::runtime::Runtime &runtime,
                                                 const infinilm::runtime::Tensor &hidden_states) const {
        if (!lm_head_) {
            throw std::runtime_error("TextCausalLMHeadR8341::logits_from_hidden called on a non-last pipeline stage");
        }
        auto lm_head_input = hidden_states;
        return lm_head_->forward(runtime, lm_head_input);
    }

    Model &model() { return *model_; }

protected:
    std::shared_ptr<Model> model_;
    std::shared_ptr<infinilm::layers::linear::DualVocabHeadR8341> lm_head_;

private:
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    size_t pp_size_{1};
    size_t pp_stage_{0};
    infinilm::runtime::Tensor last_token_shift_;
};

}
