#pragma once

#include "packed_kv_b_tp16.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "execution/submit.hpp"

namespace glm_nov_value::tp16 {

namespace rt = infinilm::runtime;
namespace ex = infinilm::execution;
namespace model = infinilm::models::glm_moe_dsa;

inline void require_tensor(rt::Runtime& r, const rt::Tensor& tensor, const rt::Shape& shape,
                           rt::DataType dtype) {
    if (!tensor || tensor->device() != r.device() || tensor->shape() != shape ||
        tensor->dtype() != dtype || !tensor->is_contiguous())
        throw std::invalid_argument("packed TP16 kv_b tensor binding differs");
}

struct PackedDeviceValue {
    const rt::Runtime* owner;
    struct { rt::Tensor weight; rt::Tensor scale; } tensors;
};

template<class Work>
PackedDeviceValue pack_device(rt::Runtime& r, const rt::Tensor& weight, const rt::Tensor& scale,
                              std::size_t global_rank, Work& work) {
    r.assertOwnerThread();
    if (r.isGraphRecording() || r.device().type() != rt::Device::Type::kHygon ||
        global_rank >= ranks || r.device().index() != int(global_rank % 8))
        throw std::invalid_argument("packed TP16 kv_b owner geometry differs");

    require_tensor(r, weight, {original_rows, columns}, rt::DataType::kUInt8);
    require_tensor(r, scale, {original_scale_rows, scale_columns}, rt::DataType::kFloat32);
    PackedDeviceValue packed{&r, {work.alloc({packed_rows, columns}, rt::DataType::kUInt8),
                                  work.alloc({tiles.size(), scale_columns}, rt::DataType::kFloat32)}};
    work.held.insert(work.held.end(), {weight, scale});
    const auto scale_base = global_rank * (original_rows / block);
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        auto source_weight = weight->narrow({{0, tiles[i] * block, block}});
        auto dest_weight = packed.tensors.weight->narrow({{0, i * block, block}});
        auto source_scale = scale->narrow({{0, scale_base + tiles[i], 1}});
        auto dest_scale = packed.tensors.scale->narrow({{0, i, 1}});
        work.held.insert(work.held.end(), {source_weight, dest_weight, source_scale, dest_scale});
        dest_weight->copy_from(r, source_weight);
        dest_scale->copy_from(r, source_scale);
    }
    r.syncStream();
    return packed;
}

struct SelectedValue {
    rt::Tensor value;
    bool cached_fp8_dispatch;
};

// The caller supplies the original validated causal IDs and a Work owner
// that retains storage through success and exception drains.
template<class Work>
SelectedValue selected(rt::Runtime& r, const PackedDeviceValue& packed,
                       const rt::Tensor& latent, const rt::Tensor& rotary,
                       const rt::Tensor& ids, Work& work) {
    r.assertOwnerThread();
    if (r.isGraphRecording() || packed.owner != &r || !ids || ids->ndim() != 1 ||
        !latent || latent->ndim() != 2)
        throw std::invalid_argument("packed TP16 kv_b selection or owner differs");
    const auto rows = ids->size(0);
    const auto cache_rows = latent->size(0);
    if (rows < 65 || rows > 2048 || cache_rows != 32768)
        throw std::invalid_argument("packed TP16 kv_b comparison requires full cache and 65..2048 selected rows");

    require_tensor(r, packed.tensors.weight, {packed_rows, columns}, rt::DataType::kUInt8);
    require_tensor(r, packed.tensors.scale, {tiles.size(), scale_columns}, rt::DataType::kFloat32);
    require_tensor(r, ids, {rows}, rt::DataType::kInt64);
    require_tensor(r, latent, {cache_rows, columns}, rt::DataType::kBFloat16);
    require_tensor(r, rotary, {cache_rows, 64}, rt::DataType::kBFloat16);
    work.held.insert(work.held.end(), {ids, latent, rotary, packed.tensors.weight, packed.tensors.scale});

    auto gathered_latent = work.alloc({rows, columns}, rt::DataType::kBFloat16);
    auto gathered_rotary = work.alloc({rows, 64}, rt::DataType::kBFloat16);
    const auto index_config = infini::ops::IndexSelect::DefaultConfig(r.device().type());
    ex::submit<infini::ops::IndexSelect>(r, index_config, latent, ids, int64_t{0}, gathered_latent);
    // Retain the rotary gather so the comparison isolates projection and output assembly.
    ex::submit<infini::ops::IndexSelect>(r, index_config, rotary, ids, int64_t{0}, gathered_rotary);

    auto bytes = work.alloc({rows, columns}, rt::DataType::kUInt8);
    auto scales = work.alloc({rows, scale_columns}, rt::DataType::kFloat32);
    auto expansion = work.alloc({rows, packed_rows}, rt::DataType::kBFloat16);
    ex::submit<infini::ops::BlockQuantF8E4m3>(
        r, ex::RecordingPolicy::HostOnly,
        infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()), gathered_latent, bytes, scales);

    const bool cached = model::fp8_cached_r809(r, expansion, bytes, scales,
                                              packed.tensors.weight, packed.tensors.scale, 0, 0);
    if (!cached)
        ex::submit<infini::ops::Fp8BlockLinear>(
            r, ex::RecordingPolicy::HostOnly,
            infini::ops::Fp8BlockLinear::DefaultConfig(r.device().type()), expansion,
            bytes, scales, packed.tensors.weight, packed.tensors.scale, std::size_t{0}, std::size_t{0});

    auto values = work.alloc({rows, heads, value_columns}, rt::DataType::kBFloat16);
    auto flat = values->view({rows, heads * value_columns});
    work.held.push_back(flat);
    for (std::size_t head = 0; head < heads; ++head) {
        auto source = expansion->narrow({{1, starts[head], value_columns}});
        auto destination = flat->narrow({{1, head * value_columns, value_columns}});
        work.held.insert(work.held.end(), {source, destination});
        destination->copy_from(r, source);
    }
    r.syncStream();
    return {values, cached};
}

}
