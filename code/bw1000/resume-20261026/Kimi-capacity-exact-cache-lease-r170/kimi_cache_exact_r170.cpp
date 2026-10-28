#include "kimi_engine_input_r145.hpp"
#include "models/kimi_k3/kimi_k3_allocate_cache.hpp"

#include "models/kimi_k3/kimi_k3_pipeline_partition.hpp"

#include "global_state/global_state.hpp"
#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include "utils.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <vector>

namespace infinilm::models::kimi_k3 {

namespace r170_cache {
size_t rank_kv_heads(infinilm::runtime::Size k_dim, infinilm::runtime::Size v_dim,
                     infinilm::runtime::Size k_heads, infinilm::runtime::Size v_heads,
                     size_t tp_size) {
    if (k_dim == 0 || v_dim == 0 || k_heads == 0 || v_heads == 0
        || k_dim != v_dim || k_heads != v_heads) {
        throw std::invalid_argument("KV cache requires positive equal K/V dimensions and head counts");
    }
    if (k_heads < tp_size) {
        if (tp_size % k_heads != 0) {
            throw std::invalid_argument("KV cache replication requires TP divisible by KV heads");
        }
        return 1;
    }
    if (k_heads % tp_size != 0) {
        throw std::invalid_argument("KV cache head count must be divisible by TP");
    }
    return k_heads / tp_size;
}
infinilm::runtime::Tensor create_exact_mla_cache(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    infinilm::runtime::Size k_dim,
    infinilm::runtime::Size v_dim,
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    infinilm::runtime::DataType dtype,
    const cache::PagedKVCacheConfig &config) {
    cache::validate_cache_runtime(runtime, rank);
    const size_t num_rank_k_heads = rank_kv_heads(k_dim, v_dim, num_k_heads, num_v_heads, static_cast<size_t>(rank.tp_size));

    size_t kv_dim = k_dim;

    size_t num_blocks_per_layer = config.num_blocks();
    size_t block_size = config.block_size();

    // Canonical FlashAttention cache layout: [K/V, blocks, block, heads, dim].
    infinilm::runtime::Shape kv_shape = {
        2, num_blocks_per_layer, block_size, num_rank_k_heads, k_dim};

    infinilm::runtime::Tensor kv_cache = infinilm::runtime::Tensor::empty_parameter(runtime, kv_shape, dtype);
    try {
        if (kv_cache->nbytes() != 0) {
            if (kv_cache->is_contiguous() && !runtime.isGraphRecording()) {
                if (!infini::ops::Fill::valid_output(infinilm::execution::detail::view(kv_cache))) {
                    throw std::invalid_argument("zero-fill output layout or storage is invalid");
                }
                runtime.setDeviceMemoryAsync(kv_cache->data(), 0, kv_cache->nbytes(), runtime.stream());
            } else {
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                    kv_cache, 0.0, kv_cache);
            }
            INFINILM_TRACE_OP(Zeros, kv_cache);
        }
    } catch (...) {
        // The factory's destination remains alive until the owner drain proves safety.
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
    // The caller retains its existing completion boundary; no per-layer sync.
    return kv_cache;
}
} // namespace r170_cache



qwen3_next::AllocatedHybridCache kimi_k3_allocate_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank_info,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    backends::AttentionBackend attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    cache::validate_cache_runtime(runtime, rank_info);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (attention_backend == backends::AttentionBackend::STATIC_ATTN) {
        throw std::runtime_error("Kimi K3 does not support static attention");
    }
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t num_heads = model_config->get<size_t>("num_attention_heads");
    const auto &linear = model_config->get_config_json().at("linear_attn_config");
    const size_t linear_head_dim = linear.at("head_dim").get<size_t>();
    const size_t linear_num_heads = linear.at("num_heads").get<size_t>();
    const size_t conv_kernel = linear.at("short_conv_kernel_size").get<size_t>();
    if (conv_kernel < 2) {
        throw std::invalid_argument("K3 conv state requires kernel >=2");
    }
    const auto kda_layers = linear.at("kda_layers").get<std::vector<size_t>>();
    std::vector<bool> is_kda(num_layers, false);
    for (const size_t one_based_layer_idx : kda_layers) {
        is_kda.at(one_based_layer_idx - 1) = true;
    }

    runtime.assertOwnerThread();
    if (!tp_size || linear_num_heads % tp_size) {
        throw std::invalid_argument("K3 state head partition invalid");
    }
    const auto [local_begin, local_end] = kimi_k3_pipeline_layer_range(
        num_layers,
        rank_info.pp_size,
        rank_info.pp_stage);
    std::vector<infinilm::runtime::Tensor> kv(num_layers);
    std::vector<infinilm::runtime::Tensor> conv(num_layers);
    std::vector<infinilm::runtime::Tensor> recurrent(num_layers);

    auto allocate_kda = [&](size_t layer_idx, size_t pool_size) {
        const size_t heads = linear_num_heads / tp_size;
        conv[layer_idx] = infinilm::runtime::Tensor::empty(runtime, {pool_size, 3 * heads * linear_head_dim, conv_kernel - 1}, model_config->get_dtype());
        recurrent[layer_idx] = infinilm::runtime::Tensor::empty(runtime, {pool_size, heads, linear_head_dim, linear_head_dim}, model_config->get_dtype());
        runtime.setDeviceMemory(conv[layer_idx]->data(), 0, conv[layer_idx]->nbytes());
        runtime.setDeviceMemory(recurrent[layer_idx]->data(), 0, recurrent[layer_idx]->nbytes());
        runtime.syncStream(); // preserve zero-row initialization before publication
    };

    const auto *config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (config == nullptr) {
        throw std::runtime_error("Kimi K3 paged attention requires PagedKVCacheConfig");
    }
    const bool compact_capability=model_config->get_config_json().value("infinilm_kimi_compact_cache_r4974",false);
    if(compact_capability)infinilm::models::kimi_native::validate_cache(config,true);
    // KDA state rows follow request capacity, independently of MLA pages per request.
    const size_t state_pool_size = compact_capability ? config->max_batch_size()+1 : std::max<size_t>(2, config->num_blocks() / 4);
    try {
        for (size_t i = local_begin; i < local_end; ++i) {
            if (is_kda[i]) {
                allocate_kda(i, state_pool_size);
            } else {
                kv[i] = (tp_size==32 && rank_info.pp_size==1 && num_layers==93 && num_heads==96 && head_dim==192 && config->num_blocks()==68 && config->block_size()==256 && model_config->get_kv_cache_dtype()==infinilm::runtime::DataType::kBFloat16)
                    ? r170_cache::create_exact_mla_cache(
                    runtime, rank_info, head_dim, head_dim, num_heads, num_heads, model_config->get_kv_cache_dtype(), *config)
                    : cache::PagedKVCache::create_layer_kv_cache(
                    runtime, rank_info,
                    head_dim, head_dim, num_heads, num_heads,
                    model_config->get_kv_cache_dtype(), *config);
            }
        }
    } catch (...) {
        // Local vectors still retain all newly allocated backing here.
        const auto primary = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); } // no unwind/reuse after a failed drain
        std::rethrow_exception(primary);
    }
    return {std::move(kv), std::move(conv), std::move(recurrent)};
}

} // namespace infinilm::models::kimi_k3
