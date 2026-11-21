#include "fresh_kv_explicit_ar.hpp"

extern "C" int modern_glm_global_fresh_kv_decode_r474(
    void* handle, size_t tokens, size_t count, const size_t* requests,
    const void* input, const void* positions, void* logits, int all_positions, void* stream) {
    try {
        auto& context = global470(handle);
        if (!context.inner->ready || all_positions != 0) {
            throw std::invalid_argument("fresh K/V requires loaded Model and final logits only");
        }
        const ::glm_r474_fresh_kv_ar::Scope phase(context.inner->owner.get(), tokens, count, requests);
        return modern_glm_global_forward_r470(handle, tokens, count, requests, input, positions, logits,
                                             all_positions, stream);
    } catch (const std::exception& error) {
        report470(error);
        return 1;
    }
}
