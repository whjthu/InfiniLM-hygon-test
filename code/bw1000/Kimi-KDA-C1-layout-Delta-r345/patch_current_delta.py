"""Only insert bounded defaultOFF event markers into exact current Delta."""
from pathlib import Path
import hashlib
BASE='cc3d6b8dd19021d758560938e46808871d3cfcc9af50868227ef43f98555b588'
def patch(text):
 assert hashlib.sha256(text.encode()).hexdigest()==BASE
 edits=[('#include "batch_state_r9122.hpp"','#include "batch_state_r9122.hpp"\n#define KIMI_DELTA_PROFILE_IMPL_R309\n#include "delta_profile_r309.hpp"'),
 ('    const auto &metadata = forward_context.mamba_metadata;','    const auto &metadata = forward_context.mamba_metadata;\n    kimi_delta_profile_r309::Session prof309(runtime,hidden_states->size(1),metadata.init_state_indices.value()->numel(),pool);'),
 ('    raw_trace.capture(qkv_projected, "qkv_projected");','    raw_trace.capture(qkv_projected, "qkv_projected");\n    prof309.mark("QKV_projection");'),
 ('    raw_trace.capture(qkv, "conv_output");','    raw_trace.capture(qkv, "conv_output");\n    prof309.mark("Conv");'),
 ('    raw_trace.capture(qkv, "silu_qkv");','    raw_trace.capture(qkv, "silu_qkv");\n    prof309.mark("Silu");'),
 ('    raw_trace.capture(beta_projected, "beta_projected");','    raw_trace.capture(beta_projected, "beta_projected");\n    prof309.mark("gate_fa_fb_beta_projections");'),
 ('    auto serial_kda_r9027=[&](){','    prof309.mark("captureOFF_and_KDA_preparation");\n    auto serial_kda_r9027=[&](){'),
 ('    raw_trace.capture(output, "kda_output");','    raw_trace.capture(output, "kda_output");\n    prof309.mark("IndexedKDA");'),
 ('    raw_trace.capture(gated, "gated");','    raw_trace.capture(gated, "gated");\n    prof309.mark("RMSNorm_gprojection_sigmoid_mul");'),
 ('    raw_trace.capture(result, "output");','    raw_trace.capture(result, "output");\n    prof309.mark("original_o_proj_and_TP_allreduce");'),
 ('    raw_trace.finish();','    raw_trace.finish();\n    prof309.finish();')]
 for before,after in edits:
  assert text.count(before)==1,before
  text=text.replace(before,after)
 # Reversal proves all original mathematical/selection/cleanup text survives.
 reverse=text
 for before,after in reversed(edits):reverse=reverse.replace(after,before)
 assert hashlib.sha256(reverse.encode()).hexdigest()==BASE
 return text
