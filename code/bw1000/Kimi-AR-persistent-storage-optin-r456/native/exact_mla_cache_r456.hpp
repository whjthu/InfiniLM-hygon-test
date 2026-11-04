#pragma once
#include "ar_storage_r456.hpp"
#include "cache/kv_cache.hpp"
#include "execution/submit.hpp"
#include "utils.hpp"
namespace kimi_ar_storage_r456 {
inline infinilm::runtime::Tensor exact_mla_cache_r456(
    infinilm::runtime::Runtime &runtime,
    const infinilm::engine::distributed::RankInfo &rank,
    infinilm::runtime::Size k_dim,
    infinilm::runtime::Size v_dim,
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    infinilm::runtime::DataType dtype,
    const infinilm::cache::PagedKVCacheConfig &config) {
    infinilm::cache::validate_cache_runtime(runtime, rank);
    kimi_ar_storage_r456::cache_geometry(config.max_batch_size(),config.num_blocks(),config.block_size());
    if(rank.tp_size!=32||rank.pp_size!=1||k_dim!=192||v_dim!=192||num_k_heads!=96||num_v_heads!=96||dtype!=infinilm::runtime::DataType::kBFloat16)throw std::invalid_argument("r456 exact MLA geometry");
    const size_t num_rank_k_heads = 3;

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
}
