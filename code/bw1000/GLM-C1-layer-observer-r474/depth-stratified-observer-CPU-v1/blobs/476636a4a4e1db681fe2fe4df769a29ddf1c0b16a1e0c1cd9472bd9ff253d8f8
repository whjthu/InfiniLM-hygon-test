#pragma once
#include "glm_sparse_observer_r131.hpp"
#include <cstdlib>

// Observation only. No new registry, State/Mark layout, event pool, or C ABI.
namespace glm_attention_detail_v1 {
template<class Runtime> inline bool enabled(Runtime& r) {
    auto* s = glm_sparse_r131::current;
    if (!s || !s->active || !s->enabled || !s->detail) return false;
    if (!glm_sparse_r131::depth_pair &&
        !((s->current_layer == 0 && s->full_layer == 0) ||
          (s->current_layer == 3 && s->shared_layer == 3))) return false;
    if (s->selector.tokens != 1 || s->selector.rows.size() != 1) return false;
    const auto& row = s->selector.rows[0];
    if (row[0] != 0 || row[1] == 0 || row[2] != 1) return false;
    void* owner = static_cast<void*>(&r);
    const auto state = glm_sparse_r131::states.find(owner);
    const auto selected = glm_sparse_r131::c1_next_decode_r474.find(owner);
    const auto frames = glm_sparse_r131::c1_total_frames_r474.find(owner);
    // Membership proves select_c1_r474; original begin consumes the explicit
    // next_decode flag and admits only a true post-prompt T1 continuation.
    const bool selected_c1 = state != glm_sparse_r131::states.end() && state->second.get() == s &&
           selected != glm_sparse_r131::c1_next_decode_r474.end() &&
           frames != glm_sparse_r131::c1_total_frames_r474.end() &&
           frames->second >= 1 && frames->second <= 3;
    if (!selected_c1) return false;
    // Independent default-OFF opt-in. Check only after the original C1 gate.
    // Missing, empty, 0, or any string other than exactly "1" is OFF, not an error.
    const char* optin = std::getenv("GLM_ATTENTION_DETAIL_V1");
    return optin && optin[0] == '1' && optin[1] == '\0';
}
template<class Runtime> inline void mark(Runtime& r, const char* name) {
    if (enabled(r)) glm_sparse_r131::detail(static_cast<cudaStream_t>(r.stream()), name);
}
template<class Runtime> inline void tuple(Runtime& r, const char* site,
        size_t m, size_t n, size_t k, size_t row_offset, size_t column_offset) {
    if (!enabled(r)) return;
    // Called only after the original BF16/contiguity guards. Raw strides are
    // NOT queried: [K,1] is canonical contiguous layout inferred from guards.
    std::string name = std::string("attnd1.") + site + ".tuple.M" + std::to_string(m) +
        ".N" + std::to_string(n) + ".K" + std::to_string(k) +
        ".row_offset" + std::to_string(row_offset) +
        ".column_offset" + std::to_string(column_offset) +
        ".input_BF16.contiguous_checked.canonical_stride_inferred_" +
        std::to_string(k) + "_1.raw_stride_UNKNOWN.kernel_provider_UNKNOWN";
    glm_sparse_r131::detail(static_cast<cudaStream_t>(r.stream()), name.c_str());
}
template<class Runtime> inline void dispatch(Runtime& r, const char* site, bool cached) {
    if (!enabled(r)) return;
    std::string name = std::string("attnd1.") + site +
        (cached ? ".dispatch.cached_r809" : ".dispatch.fallback_Fp8BlockLinear") +
        ".kernel_provider_UNKNOWN";
    glm_sparse_r131::detail(static_cast<cudaStream_t>(r.stream()), name.c_str());
}
}
