import hashlib
from patch_current_delta import patch as profile_patch
def patch(text):
 text=profile_patch(text)
 text=text.replace('#include "delta_profile_r309.hpp"','#include "delta_profile_r309.hpp"\n#include "fused_r343.hpp"')
 marker='    auto serial_kda_r9027=[&](){'
 # Insert at the existing final branch, keeping every original fallback intact.
 start=text.index(marker);end=text.index('    raw_trace.capture(output, "kda_output");',start)
 old=text[start:end]
 needle='    if ('
 # Existing selector chain starts after the serial lambda. Locate actual call to it.
 line=old.index('\n    if(serial_kda_r9027())')
 assert line>=0
 extra='''
    if (fused_r343::run(runtime,{output->view(packed_shape),forward_context.ssm_state_vec.at(layer_idx_),q_heads->view(packed_shape),k_heads->view(packed_shape),v_heads->view(packed_shape),gate_decay->view(packed_shape),beta->view({1,batch_size*seq_len,local_num_heads_}),A_log_.tensor(),dt_bias,metadata.input_offsets.value(),metadata.init_state_indices.value(),metadata.final_state_indices.value()},1.0f/std::sqrt(static_cast<float>(head_dim_)),gate_lower_bound_,true)) {
    } else'''
 new=old[:line]+extra+old[line:]
 assert new.count(extra)==1
 return text[:start]+new+text[end:]
