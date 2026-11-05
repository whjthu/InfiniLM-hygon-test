#pragma once

#include "accepted_prefix_lengths.hpp"
#include "glm_moe_dsa_projected_text_model_r22.hpp"

namespace infinilm::models::glm_moe_dsa::block_verify {

struct NativeLengthsAccess {
    // The caller must bind a successful, drained, all-positions C1 forward.
    static bool apply(TextModelProjectedR22& model, runtime::Runtime& runtime,
                      StagedPrefix prefix, bool publish) {
        model.check_owner(runtime);
        if (!model.ready_ || model.poisoned_ || model.resident_any_r5640_ ||
            model.rows_ != 1 || model.capacity_ != 32768 || model.lengths_.size() != 1 ||
            model.rank_.tp_size != 8 || model.rank_.pp_size != 2) {
            return false;
        }
        if (model.rank_.pp_stage == 0 && model.begin_ == 0 && model.end_ == 38) {
            return bind<38, 11>(model, runtime, prefix, publish);
        }
        if (model.rank_.pp_stage == 1 && model.begin_ == 38 && model.end_ == 78) {
            return bind<40, 10>(model, runtime, prefix, publish);
        }
        return false;
    }

private:
    template <std::size_t Layers, std::size_t Indexers>
    static bool bind(TextModelProjectedR22& model, runtime::Runtime& runtime,
                     StagedPrefix prefix, bool publish) {
        if (model.layers_.size() != Layers) return false;
        OwnerLengths<Layers, Indexers> lengths{&model.lengths_[0], {}, {}};
        std::size_t selectors = 0;
        for (std::size_t layer = 0; layer < Layers; ++layer) {
            const auto& decoder = model.layers_[layer];
            if (!decoder || decoder->owner_ != &runtime || !decoder->ready_ || decoder->poisoned_) {
                return false;
            }
            const auto& attention = decoder->attention_;
            if (!attention || attention->owner_ != &runtime || !attention->ready_ ||
                attention->poisoned_ || attention->rows_ != 1 ||
                attention->capacity_ != 32768 || attention->topk_ != 2048 ||
                attention->lengths_.size() != 1 || attention->host_bank_r4504_ ||
                (attention->resident_policy_r5640_ && *attention->resident_policy_r5640_) ||
                !attention->latent_ || !attention->rotary_ || !attention->projected_key_ ||
                attention->projected_value_ || attention->full_ != bool(attention->selector_)) {
                return false;
            }
            lengths.attention[layer] = &attention->lengths_[0];
            const auto& selector = attention->selector_;
            if (!selector) continue;
            if (selectors == Indexers || selector->owner_ != &runtime || selector->poisoned_ ||
                selector->rows_ != 1 || selector->capacity_ != 32768 ||
                selector->topk_ != 2048 || selector->tp_size_ != 8 || !selector->cache_ ||
                selector->lengths_.size() != 1 || selector->missing_keys_r1406_.size() != 1 ||
                selector->missing_keys_r1406_[0]) {
                return false;
            }
            lengths.selector[selectors++] = &selector->lengths_[0];
        }
        return selectors == Indexers &&
            (publish ? lengths.publish_after_owner_drain(prefix)
                     : lengths.can_publish_after_owner_drain(prefix));
    }
};

}  // namespace infinilm::models::glm_moe_dsa::block_verify
