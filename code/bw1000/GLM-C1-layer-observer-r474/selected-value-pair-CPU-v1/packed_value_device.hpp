#pragma once

#include "packed_kv_b.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "execution/submit.hpp"

namespace glm_nov_value {
namespace rt = infinilm::runtime;
namespace ex = infinilm::execution;
namespace model = infinilm::models::glm_moe_dsa;

struct DevicePacked {
    rt::Tensor weight;
    rt::Tensor scale;
};

inline void require_tensor(rt::Runtime& r, const rt::Tensor& tensor, const rt::Shape& shape,
                           rt::DataType dtype) {
    if (!tensor || tensor->device() != r.device() || tensor->shape() != shape ||
        tensor->dtype() != dtype || !tensor->is_contiguous())
        throw std::invalid_argument("packed kv_b tensor binding differs");
}

template<class Work>
DevicePacked pack_device(rt::Runtime& r, const rt::Tensor& weight, const rt::Tensor& scale,
                         std::size_t rank, Work& work) {
    r.assertOwnerThread();
    if (r.isGraphRecording() || r.device().type() != rt::Device::Type::kHygon || rank >= ranks ||
        r.device().index() != int(rank))
        throw std::invalid_argument("packed kv_b owner geometry differs");
    require_tensor(r, weight, {original_rows, columns}, rt::DataType::kUInt8);
    require_tensor(r, scale, {original_scale_rows, scale_columns}, rt::DataType::kFloat32);
    DevicePacked result{work.alloc({packed_rows, columns}, rt::DataType::kUInt8),
                        work.alloc({tiles.size(), scale_columns}, rt::DataType::kFloat32)};
    const auto base = rank * (original_rows / block);
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const auto src_w = weight->narrow({{0, tiles[i] * block, block}});
        const auto dst_w = result.weight->narrow({{0, i * block, block}});
        const auto src_s = scale->narrow({{0, base + tiles[i], 1}});
        const auto dst_s = result.scale->narrow({{0, i, 1}});
        work.held.insert(work.held.end(), {src_w, dst_w, src_s, dst_s});
        dst_w->copy_from(r, src_w);
        dst_s->copy_from(r, src_s);
    }
    r.syncStream();
    return result;
}

template<class Work>
rt::Tensor selected(rt::Runtime& r, const DevicePacked& packed, const rt::Tensor& latent,
                    const rt::Tensor& rotary, const rt::Tensor& ids, Work& work) {
    r.assertOwnerThread();
    if (r.isGraphRecording() || !ids || ids->ndim() != 1)
        throw std::invalid_argument("packed kv_b selection geometry differs");
    const auto rows = ids->size(0);
    if (rows < 65 || rows > 2048)
        throw std::invalid_argument("packed kv_b comparison requires 65..2048 selected rows");
    require_tensor(r, packed.weight, {packed_rows, columns}, rt::DataType::kUInt8);
    require_tensor(r, packed.scale, {tiles.size(), scale_columns}, rt::DataType::kFloat32);
    require_tensor(r, ids, {rows}, rt::DataType::kInt64);
    // Preserve both gathers from the baseline. Only the projection and its
    // K/V output materialization change in this comparison.
    auto l = work.alloc({rows, columns}, rt::DataType::kBFloat16);
    auto k = work.alloc({rows, 64}, rt::DataType::kBFloat16);
    auto index_config = infini::ops::IndexSelect::DefaultConfig(r.device().type());
    ex::submit<infini::ops::IndexSelect>(r, index_config, latent, ids, int64_t{0}, l);
    ex::submit<infini::ops::IndexSelect>(r, index_config, rotary, ids, int64_t{0}, k);
    auto x = work.alloc({rows, columns}, rt::DataType::kUInt8);
    auto xs = work.alloc({rows, scale_columns}, rt::DataType::kFloat32);
    auto expanded = work.alloc({rows, packed_rows}, rt::DataType::kBFloat16);
    ex::submit<infini::ops::BlockQuantF8E4m3>(
        r, ex::RecordingPolicy::HostOnly,
        infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()), l, x, xs);
    if (!model::fp8_cached_r809(r, expanded, x, xs, packed.weight, packed.scale, 0, 0))
        throw std::logic_error("packed kv_b mode1 dispatch declined");
    auto values = work.alloc({rows, heads, value_columns}, rt::DataType::kBFloat16);
    auto flat_values = values->view({rows, heads * value_columns});
    work.held.push_back(flat_values);
    for (std::size_t head = 0; head < heads; ++head) {
        auto src = expanded->narrow({{1, starts[head], value_columns}});
        auto dst = flat_values->narrow({{1, head * value_columns, value_columns}});
        work.held.insert(work.held.end(), {src, dst});
        dst->copy_from(r, src);
    }
    r.syncStream();
    return values;
}
}
