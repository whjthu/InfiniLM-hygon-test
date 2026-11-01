"""DefaultOFF one-allocation candidate applied to exact current Delta source."""
import hashlib
BASE='cc3d6b8dd19021d758560938e46808871d3cfcc9af50868227ef43f98555b588'


def patch(text):
    assert hashlib.sha256(text.encode()).hexdigest()==BASE
    before='#include "batch_state_r9122.hpp"'
    after=before+'\n#define KIMI_GATE_ALIAS_IMPL_R314\n#include "gate_alias_r314.hpp"'
    assert text.count(before)==1;text=text.replace(before,after)
    old='    auto output_gate = infinilm::runtime::Tensor::empty((*forward_context.runtime), output_gate_input->shape(), output_gate_input->dtype());'
    new='''    // The g_proj result is fresh. Its only subsequent mathematical consumer
    // is Sigmoid, then Mul. Trace-enabled paths retain the original allocation.
    const auto trace_requested_r314=[](){
        for(const char*name:{"INFINILM_KIMI_TRACE_DIR","INFINILM_KIMI_KDA_TRACE_DIR",
              "INFINILM_KIMI_KDA_TRACE_LAYER","INFINILM_KIMI_TARGET_TAPS_DIR_R9022"}){
            const char*v=std::getenv(name);if(v&&*v)return true;
        }
        return false;
    };
    const bool alias_r314=kimi_gate_alias_r314::enabled(runtime)
        && runtime.graphCaptureState()==infinilm::execution::graph::GraphManager::CaptureState::kInactive
        && runtime.device().type()==infinilm::runtime::Device::Type::kHygon
        && output_gate_input->dtype()==infinilm::runtime::DataType::kBFloat16
        && output_gate_input->shape()==infinilm::runtime::Shape{1,1,768}
        && metadata.init_state_indices.value()->numel()==1 && pool==3
        && infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==0
        && !trace_requested_r314();
    kimi_gate_alias_r314::hit(runtime,alias_r314);
    auto output_gate = alias_r314 ? output_gate_input : infinilm::runtime::Tensor::empty((*forward_context.runtime), output_gate_input->shape(), output_gate_input->dtype());'''
    assert text.count(old)==1;text=text.replace(old,new)
    reverse=text.replace(new,old).replace(after,before)
    assert hashlib.sha256(reverse.encode()).hexdigest()==BASE
    return text
