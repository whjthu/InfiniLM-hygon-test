#include "fresh_kv_prefill512.hpp"

extern "C" int modern_glm_global_fresh_kv_prefill512_r474(
    void* handle, size_t tokens, size_t count, const size_t* requests,
    const void* input, const void* positions, void* logits, int all_positions, void* stream,
    size_t* evidence, size_t evidence_words) {
    namespace prefill = ::glm_r474_fresh_kv_prefill512;
    try {
        if (!evidence || evidence_words != prefill::sections) {
            throw std::invalid_argument("fresh K/V T512 evidence buffer differs");
        }
        for (size_t i = 0; i < evidence_words; ++i) {
            evidence[i] = 0;
        }
        auto& context = global470(handle);
        if (!context.inner->ready || all_positions != 0 || !stream) {
            throw std::invalid_argument("fresh K/V T512 requires loaded Model, owned stream and final logits");
        }
        prefill::Scope phase(context.inner->owner.get(), tokens, count, requests);
        const int rc = modern_glm_global_forward_r470(handle, tokens, count, requests, input, positions, logits,
                                                     all_positions, stream);
        phase.publish(evidence);
        return rc;
    } catch (const std::exception& error) {
        report470(error);
        return 1;
    }
}
