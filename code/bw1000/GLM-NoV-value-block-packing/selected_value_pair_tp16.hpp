#pragma once

#include "packed_value_device_tp16.hpp"
#include "selected_values_r400.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <optional>

namespace glm_nov_value::tp16 {

enum class SelectedArm { original, packed };

struct SelectedPhase {
    rt::Tensor value;
    std::chrono::nanoseconds elapsed;
    // The baseline records no provider branch in the existing producer.
    std::optional<bool> packed_cache_dispatch;
};

struct SelectedPairBinding {
    const model::AttentionPrepare* original;
    PackedDeviceValue packed;
};

// Bind the original kv_b bytes once, before either arm enters its timed window.
template<class Work>
SelectedPairBinding bind_selected_pair(model::AttentionPrepare& prepare, rt::Runtime& runtime,
                                     std::size_t global_rank, Work& work) {
    if (prepare.local_heads() != heads)
        throw std::invalid_argument("selected-value pair requires TP16 attention");
    auto weights = prepare.state_dict();
    return {&prepare, pack_device(runtime, weights.at("kv_b_proj.weight").tensor(),
                                 weights.at("kv_b_proj.weight_scale_inv").tensor(), global_rank, work)};
}

// Both arms take the caller's original causal IDs and use the same outer Work
// lifetime. Setup, tensor readback, and comparison happen outside this window.
template<class Work>
SelectedPhase run_selected_phase(SelectedArm arm, model::AttentionPrepare& prepare,
                                 rt::Runtime& runtime, const SelectedPairBinding& binding,
                                 const rt::Tensor& latent, const rt::Tensor& rotary,
                                 const rt::Tensor& ids, Work& work) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || binding.packed.owner != &runtime || binding.original != &prepare ||
        prepare.local_heads() != heads ||
        !ids || ids->ndim() != 1 || ids->size(0) < 65 || ids->size(0) > 2048)
        throw std::invalid_argument("selected-value pair geometry or owner differs");
    require_tensor(runtime, latent, {32768, columns}, rt::DataType::kBFloat16);
    require_tensor(runtime, rotary, {32768, 64}, rt::DataType::kBFloat16);
    require_tensor(runtime, ids, {ids->size(0)}, rt::DataType::kInt64);
    work.held.insert(work.held.end(), {latent, rotary, ids});

    // Isolate the selected-value phase from earlier work enqueued by the caller.
    runtime.syncStream();
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    rt::Tensor output;
    std::optional<bool> cached;
    switch (arm) {
    case SelectedArm::original:
        output = model::v_on_demand_r400::selected_values(prepare, runtime, latent, rotary, ids, work.held);
        break;
    case SelectedArm::packed: {
        auto result = selected(runtime, binding.packed, latent, rotary, ids, work);
        output = std::move(result.value);
        cached = result.cached_fp8_dispatch;
        break;
    }
    default:
        throw std::invalid_argument("unknown selected-value comparison arm");
    }
    // Both producer calls have already drained their stream before returning.
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started);
    require_tensor(runtime, output, {ids->size(0), heads, value_columns}, rt::DataType::kBFloat16);
    return {std::move(output), elapsed, cached};
}


struct SelectedValueDifference {
    std::size_t elements;
    std::size_t different_bits;
    std::size_t nonfinite_original;
    std::size_t nonfinite_packed;

    bool exact_finite() const noexcept {
        return different_bits == 0 && nonfinite_original == 0 && nonfinite_packed == 0;
    }
};

// Validate the entire V buffer after the independently timed calls.
inline SelectedValueDifference compare_selected_values(rt::Runtime& runtime,
                                                        const SelectedPhase& original,
                                                        const SelectedPhase& packed) {
    runtime.assertOwnerThread();
    const auto& value = original.value;
    if (runtime.isGraphRecording() || !value || value->ndim() != 3 ||
        value->size(0) < 65 || value->size(0) > 2048 ||
        original.packed_cache_dispatch.has_value() || !packed.packed_cache_dispatch.has_value())
        throw std::invalid_argument("selected-value comparison arms or rows differ");
    const rt::Shape shape{value->size(0), heads, value_columns};
    require_tensor(runtime, value, shape, rt::DataType::kBFloat16);
    require_tensor(runtime, packed.value, shape, rt::DataType::kBFloat16);

    const auto count = value->size(0) * heads * value_columns;
    std::vector<std::uint16_t> a(count), b(count);
    try {
        runtime.memcpyD2H(a.data(), value->data(), count * sizeof(a[0]));
        runtime.memcpyD2H(b.data(), packed.value->data(), count * sizeof(b[0]));
        runtime.syncStream();
    } catch (...) {
        // Keep both host buffers and the caller's tensors alive through the drain.
        try { runtime.syncStream(); }
        catch (...) { std::terminate(); }
        throw;
    }
    SelectedValueDifference result{count, 0, 0, 0};
    for (std::size_t i = 0; i < count; ++i) {
        result.different_bits += a[i] != b[i];
        result.nonfinite_original += (a[i] & 0x7f80u) == 0x7f80u;
        result.nonfinite_packed += (b[i] & 0x7f80u) == 0x7f80u;
    }
    return result;
}

}
