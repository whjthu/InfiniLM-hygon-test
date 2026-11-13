#include "projection_host_timing.hpp"

#include <cassert>
#include <stdexcept>

using namespace glm_projection_host_timing;
using namespace std::chrono_literals;

// Diagnostic C ABI. The component caller supplies the actual Runtime identity and
// calls both endpoints from its owner thread, outside forward. Each row has 9 words.
extern "C" int modern_glm_projection_host_timing_begin(void* runtime) noexcept {
    return glm_projection_host_timing::capture.begin(runtime) ? 0 : -1;
}

extern "C" int modern_glm_projection_host_timing_end(
    void* runtime, std::uint64_t* rows, std::size_t capacity_rows, std::size_t* count,
    int* dropped) noexcept {
    auto& capture = glm_projection_host_timing::capture;
    if (!runtime || capture.owner != runtime || !count || !dropped ||
        (capture.size && (!rows || capacity_rows < capture.size))) return -1;
    for (std::size_t i = 0; i < capture.size; ++i) {
        const auto& sample = capture.samples[i];
        const std::array<std::uint64_t, 9> values{
            sample.rows, sample.columns, sample.tokens, sample.cached, sample.lease_active,
            sample.fence_ns, sample.return_cleanup_ns, sample.next_quant_gap_ns,
            sample.next_quant_observed};
        for (std::size_t column = 0; column < values.size(); ++column) {
            rows[i * values.size() + column] = values[column];
        }
    }
    *count = capture.size;
    *dropped = capture.dropped;
    return capture.end(runtime) ? 0 : -1;
}

int main() {
    {
        ProjectionTiming unarmed(nullptr, 1, 1, 1);
        assert(!unarmed.enabled());
    }
    int timed = 0;
    assert(modern_glm_projection_host_timing_begin(&timed)==0);
    assert(modern_glm_projection_host_timing_begin(&timed)==-1);
    bool cleaned = false;
    struct Cleanup {
        bool& cleaned;
        ~Cleanup() { cleaned = true; }
    };
    {
        ProjectionTiming timing(&timed, 2048, 6144, 1);
        Cleanup cleanup{cleaned};
        timing.before_fence(true, false);
        timing.after_fence();
    }
    std::size_t count = 0;
    int dropped = -1;
    std::uint64_t values[9]{};
    assert(cleaned);
    assert(modern_glm_projection_host_timing_end(&timed,values,0,&count,&dropped)==-1);
    assert(modern_glm_projection_host_timing_end(&timed,values,1,&count,&dropped)==0);
    assert(count==1 && dropped==0 && values[0]==2048 && values[1]==6144);
    assert(values[2]==1 && values[3]==1 && values[4]==0 && values[8]==0);
    assert(modern_glm_projection_host_timing_end(&timed,values,1,&count,&dropped)==-1);
    int first = 0, foreign = 0;
    assert(!capture.begin(nullptr));
    assert(capture.begin(&first));
    assert(!capture.begin(&first));
    assert(!capture.begin(&foreign));
    assert(!capture.end(&foreign));
    {
        ProjectionTiming disabled(&foreign, 1, 1, 1);
        assert(!disabled.enabled());
        disabled.before_quant();
        disabled.after_fence();
    }
    assert(capture.size == 0);
    {
        ProjectionTiming timing(&first, 1024, 2048, 1);
        timing.before_quant();
        timing.before_fence(true, false);
        timing.after_fence();
    }
    assert(capture.size == 1 && capture.waiting_for_quant);
    assert(capture.samples[0].cached && !capture.samples[0].lease_active);
    const auto returned = capture.last_return;
    capture.before_quant(&foreign, returned + 17ns);
    assert(!capture.samples[0].next_quant_observed);
    capture.before_quant(&first, returned + 17ns);
    assert(capture.samples[0].next_quant_observed && capture.samples[0].next_quant_gap_ns == 17);
    try {
        ProjectionTiming failed(&first, 1024, 2048, 1);
        failed.before_fence(false, true);
        failed.after_fence();
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error&) {
    }
    assert(capture.size == 1);
    for (std::size_t i = 1; i < Capture::capacity + 1; ++i) {
        ProjectionTiming timing(&first, 1792, 512, 1);
        timing.before_quant();
        timing.before_fence(false, true);
        timing.after_fence();
    }
    assert(capture.size == Capture::capacity && capture.dropped);
    assert(capture.samples[Capture::capacity - 1].next_quant_observed);
    assert(capture.samples[Capture::capacity - 1].lease_active);
    assert(capture.end(&first));
    assert(capture.begin(&first));
    assert(capture.size == 0 && !capture.waiting_for_quant && !capture.dropped);
    {
        ProjectionTiming interrupted(&first, 1792, 512, 1);
        interrupted.before_fence(false, false);
    }
    assert(capture.size == 0);
    assert(capture.end(&first));
}
