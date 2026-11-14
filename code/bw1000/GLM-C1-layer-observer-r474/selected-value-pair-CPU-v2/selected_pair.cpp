#include "selected_value_pair_tp16.hpp"

namespace pair = glm_nov_value::tp16;
namespace rt = infinilm::runtime;
using Prepare = infinilm::models::glm_moe_dsa::AttentionPrepare;

namespace {
struct Work {
    rt::Runtime& runtime;
    std::vector<rt::Tensor> held;

    ~Work() noexcept {
        try { runtime.syncStream(); }
        catch (...) { std::terminate(); }
    }

    rt::Tensor alloc(rt::Shape shape, rt::DataType dtype) {
        auto tensor = rt::Tensor::empty(runtime, shape, dtype);
        held.push_back(tensor);
        return tensor;
    }
};
}

// The owner supplies its live, loaded AttentionPrepare and validated causal
// indices. Pack once, outside the two measured intervals.
extern "C" int glm_tp16_selected_pair_create(
    Prepare* prepare, rt::Runtime* runtime, std::size_t rank,
    pair::SelectedPairBinding** binding) noexcept {
    if (!prepare || !runtime || !binding || *binding)
        return 1;
    try {
        Work work{*runtime, {}};
        *binding = new pair::SelectedPairBinding(pair::bind_selected_pair(*prepare, *runtime, rank, work));
        return 0;
    } catch (...) { return 2; }
}

extern "C" int glm_tp16_selected_pair_measure(
    Prepare* prepare, rt::Runtime* runtime, const pair::SelectedPairBinding* binding,
    const rt::Tensor* latent, const rt::Tensor* rotary, const rt::Tensor* ids,
    int packed_first, std::uint64_t* report, std::size_t report_size) noexcept {
    if (!prepare || !runtime || !binding || !latent || !rotary || !ids ||
        !report || report_size != 7 || (packed_first != 0 && packed_first != 1))
        return 1;
    try {
        const auto measure = [&](pair::SelectedArm arm) {
            Work work{*runtime, {}};
            return pair::run_selected_phase(arm, *prepare, *runtime, *binding, *latent, *rotary, *ids, work);
        };
        const auto first = packed_first ? pair::SelectedArm::packed : pair::SelectedArm::original;
        const auto second = packed_first ? pair::SelectedArm::original : pair::SelectedArm::packed;
        const auto a = measure(first);
        const auto b = measure(second);
        const auto& original = packed_first ? b : a;
        const auto& packed = packed_first ? a : b;
        const auto difference = pair::compare_selected_values(*runtime, original, packed);
        report[0] = original.elapsed.count();
        report[1] = packed.elapsed.count();
        report[2] = difference.elements;
        report[3] = difference.different_bits;
        report[4] = difference.nonfinite_original;
        report[5] = difference.nonfinite_packed;
        report[6] = packed.packed_cache_dispatch.value();
        return difference.exact_finite() ? 0 : 3;
    } catch (...) { return 2; }
}

extern "C" int glm_tp16_selected_pair_destroy(
    rt::Runtime* runtime, pair::SelectedPairBinding* binding) noexcept {
    if (!runtime || !binding || binding->packed.owner != runtime)
        return 1;
    try {
        runtime->assertOwnerThread();
        runtime->syncStream();
        delete binding;
        return 0;
    } catch (...) { return 2; }
}
