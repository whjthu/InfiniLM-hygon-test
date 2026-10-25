#include <cstdlib>
#include <array>
#include <atomic>
#include "kimi_spec_math_r3676.hpp"

#include "/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/csrc/engine/distributed/kimi_reduce_precision_r3226.hpp"
#include "kimi_owned_indices_r1213.hpp"
#include "kimi_k3_kda_trace.hpp"
#include "execution/submit.hpp"
#include "engine/distributed/communication_group.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "kimi_k3_delta_attention.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "execution/submit.hpp"

#include "../../global_state/global_state.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"

#include <runtime/context.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <runtime/context.hpp>
#include <stdexcept>
#include <string>
#include <utility>

#include <cstdint>
namespace {thread_local std::array<std::uint64_t,3>kimi_rowar_counts_r9043{};}
extern "C" int modern_kimi_rowar_stats_r9043(std::uint64_t*out,std::size_t count){if(!out||count!=3)return 1;for(std::size_t j=0;j<3;++j)out[j]=kimi_rowar_counts_r9043[j];return 0;}
namespace infinilm::models::kimi_k3 {
inline std::array<std::array<std::atomic<uint64_t>,3>,8> serial_kda_counts_r9027{};
extern "C" int modern_kimi_KDA_serial_counts_r9027(int device,uint64_t*out,size_t n){
    if(device<0||device>=8||!out||n!=3)return 1;
    for(size_t i=0;i<3;++i)out[i]=serial_kda_counts_r9027[device][i].load();return 0;
}


inline std::array<std::array<std::atomic<uint64_t>,3>,8> ccl512_counts_r8603{};
extern "C" int modern_kimi_CCL512_counts_r8603(int device,uint64_t*out,size_t n){
    if(device<0||device>=8||!out||n!=3)return 1;
    for(size_t i=0;i<3;++i)out[i]=ccl512_counts_r8603[device][i].load();return 0;
}
// Default-OFF scoped uniform512 prefill candidate; no weight/layout changes.
inline std::array<std::array<std::atomic<uint64_t>,3>,8> prefill512_counts_r8562{};
extern "C" int modern_kimi_prefill512_counts_r8562(int device,uint64_t*out,size_t n){
 if(device<0||device>=8||!out||n!=3)return 1;
 for(size_t i=0;i<3;++i)out[i]=prefill512_counts_r8562[device][i].load();return 0;
}
inline std::array<std::array<std::atomic<uint64_t>,9>,8> gate_projection_counts_r4576{};
extern "C" int modern_kimi_KDA_gate_counts_r4576(int device,uint64_t*out,size_t n){
 if(device<0||device>=8||!out||n!=9)return 1;
 for(size_t i=0;i<9;++i)out[i]=gate_projection_counts_r4576[device][i].load();return 0;
}
inline std::array<std::array<std::atomic<uint64_t>,3>,8> input_projection_counts_r4305{};
extern "C" int modern_kimi_KDA_input_counts_r4305(int device,uint64_t*out,size_t n){
 if(device<0||device>=8||!out||n!=3)return 1;
 for(size_t i=0;i<3;++i)out[i]=input_projection_counts_r4305[device][i].load();return 0;
}

inline std::array<std::array<std::atomic<uint64_t>,3>,8> projection_counts_r3932{};
extern "C" int modern_kimi_projection_counts_r3932(int device,uint64_t*out,size_t size){if(device<0||device>=8||!out||size!=3)return 1;for(size_t i=0;i<3;++i)out[i]=projection_counts_r3932[device][i].load();return 0;}


namespace {

// Private actual constructed type; KDA header/base pointer layout is unchanged.
class Prefill512RowR8603 final : public infinilm::layers::linear::RowParallelLinear {
public:
    using infinilm::layers::linear::RowParallelLinear::RowParallelLinear;
    infinilm::runtime::Tensor forward_rowar_r9043(infinilm::runtime::Runtime &runtime,infinilm::runtime::Tensor &input,int mode)const{
      runtime.assertOwnerThread();if(runtime.isGraphRecording()||tp_size_!=16||!communicator_||input->dtype()!=infinilm::runtime::DataType::kBFloat16||input->ndim()!=3||input->size(0)!=1||input->size(2)!=768||(input->size(1)!=1&&input->size(1)!=8)||has_bias()||alpha()!=1.f||output_dtype()!=input->dtype()||mode<0||mode>2||infinilm::engine::distributed::reduce_precision_r3226::selected(runtime)!=0)throw std::invalid_argument("9043 bounded original BF16 TP16 o_proj");
      std::vector<infinilm::runtime::Tensor>held,parts;infinilm::runtime::Tensor output;
      struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
      if(mode==2&&input->size(1)==8){output=infinilm::runtime::Tensor::empty(runtime,{1,8,7168},input->dtype());for(size_t j=0;j<8;++j){auto row=input->narrow({{1,j,1}});held.push_back(row);auto part=compute_linear(runtime,row);held.push_back(part);if(part->shape()!=infinilm::runtime::Shape{1,1,7168})throw std::logic_error("9043 canonical W1 shape");output->narrow({{1,j,1}})->copy_from(runtime,part);}}
      else output=compute_linear(runtime,input);
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9042_o_proj_local");
      if(mode&&input->size(1)==8){
        for(size_t j=0;j<8;++j){auto part=infinilm::runtime::Tensor::empty(runtime,{1,1,7168},output->dtype());part->copy_from(runtime,output->narrow({{1,j,1}}));if(!part->is_contiguous()||part->numel()!=7168)throw std::logic_error("9043 packed row");parts.push_back(part);}
        // Eight independent original messages. No external group, no per-row stream wait.
        for(auto &part:parts){infinilm::engine::distributed::allreduce_(runtime,part,part,infinicclSum,communicator_);++kimi_rowar_counts_r9043[2];}
        ++kimi_rowar_counts_r9043[mode-1];
        for(size_t j=0;j<8;++j)output->narrow({{1,j,1}})->copy_from(runtime,parts[j]);
      }else infinilm::engine::distributed::allreduce_(runtime,output,output,infinicclSum,communicator_);
      infinilm::models::kimi_k3::trace_kda_projection(runtime,output,"r9042_o_proj_reduced");return output;
    }
    infinilm::runtime::Tensor forward_prefill512(
        infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) const {
        // Unsupported scopes use the original actual provider unchanged.
        if (tp_size_ != 8 || !communicator_ || runtime.isGraphRecording()
            || runtime.device().type() != infinilm::runtime::Device::Type::kHygon
            || input->dtype() != infinilm::runtime::DataType::kBFloat16
            || input->ndim() != 3 || input->size(0) != 1
            || input->size(1) <= 512 || input->size(1) > 8192
            || input->size(1) % 512 || has_bias() || alpha() != 1.0F
            || output_dtype() != input->dtype()) {
            return infinilm::layers::linear::RowParallelLinear::forward(runtime,input);
        }
        runtime.assertOwnerThread();
        auto output = compute_linear(runtime,input); // Original fullM local GEMM.
        const size_t tokens=input->size(1),width=output->shape().back();
        if(width!=7168 || !output->is_contiguous())
            throw std::logic_error("8603 actual contiguous KDA output geometry");
        auto matrix=output->view({tokens,width});
        if(matrix->data()!=output->data() || !matrix->is_contiguous()
            || matrix->strides().size()!=2 || matrix->strides()[0]!=width
            || matrix->strides()[1]!=1)
            throw std::logic_error("8603 zero-copy KDA output view");
        for(size_t pos=0;pos<tokens;pos+=512){
            auto part=matrix->narrow({{0,pos,512}});
            if(!part->is_contiguous() || part->data()!=output->data()+pos*width*2)
                throw std::logic_error("8603 actual zero-copy512 row slice");
            infinilm::engine::distributed::allreduce_(runtime,part,part,infinicclSum,communicator_);
        }
        auto &counts=ccl512_counts_r8603.at(runtime.device().index());
        counts[0].fetch_add(1);counts[1].fetch_add(tokens);counts[2].fetch_add(tokens/512);
        return output;
    }
};

void checked_construction_rank(infinilm::runtime::Runtime &runtime,
                               const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("K3 construction requires inactive graph state");
    }
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size) {
        throw std::invalid_argument("K3 construction rank/device differs");
    }
}

// Match existing Parameter physical stride/byte bounds before any child allocates.
void checked_parameter_geometry(std::initializer_list<size_t> shape,
                                const infinilm::runtime::DataType &dtype) {
    size_t elements = 1;
    const auto max_stride = static_cast<size_t>(std::numeric_limits<infinilm::runtime::Stride>::max());
    for (auto it = std::rbegin(shape); it != std::rend(shape); ++it) {
        if (!*it || elements > max_stride / *it) {
            throw std::length_error("K3 construction parameter geometry overflows");
        }
        elements *= *it;
    }
    const auto bytes = infinilm::runtime::dsize(dtype);
    if (!bytes || elements > std::numeric_limits<size_t>::max() / bytes) {
        throw std::length_error("K3 construction parameter byte size overflows");
    }
}
} // namespace

KimiK3ShortConv::KimiK3ShortConv(infinilm::runtime::Runtime &runtime,
                                 size_t full_channels,
                                 size_t kernel_size,
                                 const infinilm::runtime::DataType &dtype,
                                 const engine::distributed::RankInfo &rank_info) {
    checked_construction_rank(runtime, rank_info);
    const auto tp_size = static_cast<size_t>(rank_info.tp_size);
    if (!full_channels || !kernel_size) {
        throw std::invalid_argument("KimiK3ShortConv: channels/kernel must be positive");
    }
    if (full_channels % tp_size != 0) {
        throw std::runtime_error("KimiK3ShortConv: channels must be divisible by tp_size");
    }
    checked_parameter_geometry({full_channels / tp_size, 1, kernel_size}, dtype);
    device_ = runtime.device();
    try {
        weight_ = infinilm::weights::Weight(runtime,
                                            {full_channels, 1, kernel_size}, dtype,
                                            0, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("weight", weight_);
    } catch (...) {
        const auto failure = std::current_exception();
        // weight_ and registered aliases remain alive in this ordinary body.
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        std::rethrow_exception(failure);
    }
}

KimiK3DeltaAttention::KimiK3DeltaAttention(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info)
    : layer_idx_(layer_idx) {
    checked_construction_rank(runtime, rank_info);
    if (!model_config) {
        throw std::invalid_argument("KimiK3DeltaAttention: model config is null");
    }
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto &linear_config = model_config->get_config_json().at("linear_attn_config");
    const size_t num_heads = linear_config.at("num_heads").get<size_t>();
    head_dim_ = linear_config.at("head_dim").get<size_t>();
    conv_kernel_size_ = linear_config.at("short_conv_kernel_size").get<size_t>();
    gate_lower_bound_ = linear_config.value("gate_lower_bound", -5.0f);
    const double eps = model_config->get<double>("rms_norm_eps");
    const auto tp_size = static_cast<size_t>(rank_info.tp_size);
    if (!hidden_size || !num_heads || !head_dim_ || head_dim_ > 128 || conv_kernel_size_ != 4 || !std::isfinite(gate_lower_bound_) || gate_lower_bound_ > 0 || !std::isfinite(eps) || eps <= 0) {
        throw std::invalid_argument("KimiK3DeltaAttention: invalid construction geometry/config");
    }
    if (num_heads % tp_size != 0) {
        throw std::runtime_error("KimiK3DeltaAttention: num_heads must be divisible by tp_size");
    }
    if (rank_info.tp_size > 1 && !rank_info.comm) {
        throw std::invalid_argument("KimiK3DeltaAttention: TP communicator is missing");
    }
    const auto max = std::numeric_limits<size_t>::max();
    if (num_heads > max / head_dim_ || num_heads * head_dim_ > max / 3) {
        throw std::length_error("KimiK3DeltaAttention: projection size overflows");
    }
    local_num_heads_ = num_heads / tp_size;
    local_projection_size_ = local_num_heads_ * head_dim_;
    const size_t projection_size = num_heads * head_dim_;
    // QKV, ShortConv, replicated/column/row Linear and scalar/norm layouts.
    // Every eventual allocation is preflighted before the first QKV child.
    checked_parameter_geometry({3 * local_projection_size_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, 1, conv_kernel_size_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({3 * local_projection_size_, 1, conv_kernel_size_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({local_num_heads_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({head_dim_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, head_dim_}, dtype);
    checked_parameter_geometry({local_projection_size_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({local_num_heads_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, hidden_size}, dtype);
    checked_parameter_geometry({head_dim_}, dtype);
    checked_parameter_geometry({hidden_size, local_projection_size_}, dtype);
    device_ = runtime.device();
    conv_dtype_ = infinilm::runtime::DataType::kFloat32;

    try {
        auto register_fn = [this](const std::string &name, infinilm::weights::Weight parameter) {
            this->bind_weight(name, std::move(parameter));
        };
        qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
            runtime, hidden_size, head_dim_, num_heads, num_heads,
            "q_proj", "k_proj", "v_proj", register_fn, nullptr, false, dtype, rank_info);
        // Retain each constructed child in a member BEFORE registration: an
        // bind_child allocation failure cannot destroy it before the drain.
        q_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, infinilm::runtime::DataType::kFloat32, rank_info);
        this->bind_child("q_conv1d", q_conv1d_);
        k_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, infinilm::runtime::DataType::kFloat32, rank_info);
        this->bind_child("k_conv1d", k_conv1d_);
        v_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, infinilm::runtime::DataType::kFloat32, rank_info);
        this->bind_child("v_conv1d", v_conv1d_);
        // The loaded pack is still allocated only by supplied-Runtime postload.
        A_log_ = infinilm::weights::Weight(runtime,
                                           {num_heads}, infinilm::runtime::DataType::kFloat32,
                                           0, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("A_log", A_log_);
        f_a_proj_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, hidden_size, head_dim_, false, dtype);
        this->bind_child("f_a_proj", f_a_proj_);
        f_b_proj_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, head_dim_, projection_size, false, dtype,
            rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("f_b_proj", f_b_proj_);
        dt_bias_ = infinilm::weights::Weight(runtime,
                                             {projection_size}, infinilm::runtime::DataType::kFloat32,
                                             0, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("dt_bias", dt_bias_);
        b_proj_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, num_heads, false, dtype,
            rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("b_proj", b_proj_);
        g_proj_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, projection_size, false, dtype,
            rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("g_proj", g_proj_);
        o_norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, eps, infinilm::runtime::DataType::kFloat32);
        this->bind_child("o_norm", o_norm_);
        o_proj_ = std::make_shared<Prefill512RowR8603>(
            runtime, projection_size, hidden_size, false, dtype,
            rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);
    } catch (...) {
        const auto failure = std::current_exception();
        // All constructed members/registered aliases are still alive here.
        // Leaf constructor failures keep their existing same-owner contract.
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        std::rethrow_exception(failure);
    }
}

void KimiK3DeltaAttention::process_weights_after_loading(
    infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_) {
        throw std::invalid_argument("KimiK3DeltaAttention: postload Runtime device differs");
    }
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("KimiK3DeltaAttention: postload requires inactive graph state");
    }
    if (postload_attempted_) {
        throw std::logic_error("KimiK3DeltaAttention: postload is single-use after mutation attempt");
    }
    if (!local_projection_size_ || !conv_kernel_size_ || local_projection_size_ > std::numeric_limits<size_t>::max() / 3) {
        throw std::invalid_argument("KimiK3DeltaAttention: invalid convolution pack geometry");
    }

    const std::array<infinilm::runtime::Tensor, 3> weights{
        q_conv1d_->weight(),
        k_conv1d_->weight(),
        v_conv1d_->weight(),
    };
    // All three operands are validated before hidden QKV mutation or allocation.
    for (const auto &weight : weights) {
        if (!weight || weight->shape() != infinilm::runtime::Shape{local_projection_size_, 1, conv_kernel_size_}) {
            throw std::runtime_error(
                "KimiK3DeltaAttention: unexpected short convolution weight shape");
        }
        if (weight->device() != runtime.device() || weight->dtype() != conv_dtype_) {
            throw std::invalid_argument("KimiK3DeltaAttention: convolution weight device/dtype differs");
        }
    }
    // BaseLinear/FusedLinear catch-side drains protect their local publication.
    // Retain the pre-transform bindings also across their successful return:
    // eager copies do not capture Tensor owners and the worker fence is later.
    auto old_qkv_parameters = qkv_proj_->state_dict();
    auto old_qkv_quantization = qkv_proj_->get_quantization();
    auto old_pack = packed_conv_weight_;
    infinilm::runtime::Tensor pending;
    std::array<infinilm::runtime::Tensor, 3> slices;
    postload_attempted_ = true;
    try {
        // Only this hidden object is owner-finalized; registered children have
        // already been finalized by the children-first model traversal.
        // Projection finalization is owned by loading_children traversal.
        pending = infinilm::runtime::Tensor::empty(
            runtime, {3 * local_projection_size_, 1, conv_kernel_size_}, conv_dtype_);
        for (size_t i = 0; i < slices.size(); ++i) {
            slices[i] = pending->narrow({{0, i * local_projection_size_, local_projection_size_}});
        }
        for (size_t i = 0; i < weights.size(); ++i) {
            slices[i]->copy_from(runtime, weights[i]);
        }
        // Finite same-owner stream fence, once at this postload boundary.
        // Sources, old/new QKV bindings, old pack, pack and all slices live here.
        runtime.syncStream();
        packed_conv_weight_ = std::move(pending);
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        // Preserve the first error. Hidden aliases may already be transformed;
        // the worker must remain Failed and this owner cannot be retried.
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor KimiK3DeltaAttention::forward(
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context) const {
    forward_context.require_runtime().assertOwnerThread();
    auto &runtime = forward_context.require_runtime();
    KdaTraceBatch raw_trace(runtime, forward_context, layer_idx_);
    if (runtime.device().type() != infinilm::runtime::Device::Type::kHygon)
        throw std::runtime_error("KimiK3 indexed KDA provider is currently qualified for Hygon only");
    if (!hidden_states || hidden_states->ndim() != 3 || hidden_states->size(0) != 1 || hidden_states->device() != runtime.device())
        throw std::invalid_argument("Kimi KDA requires same-owner packed [1,tokens,hidden] input");
    const auto &conv_preflight = forward_context.conv_state_vec.at(layer_idx_);
    const auto &state_preflight = forward_context.ssm_state_vec.at(layer_idx_);
    if (!conv_preflight || !state_preflight || state_preflight->ndim() != 4)
        throw std::invalid_argument("Kimi KDA state pools are missing");
    const size_t pool = state_preflight->size(0);
    if (!pool || state_preflight->shape() != infinilm::runtime::Shape{pool, local_num_heads_, head_dim_, head_dim_} || conv_preflight->shape() != infinilm::runtime::Shape{pool, 3 * local_projection_size_, 3} || state_preflight->dtype() != hidden_states->dtype() || conv_preflight->dtype() != hidden_states->dtype() || state_preflight->device() != runtime.device() || conv_preflight->device() != runtime.device())
        throw std::invalid_argument("Kimi KDA state pool geometry/dtype/device differs");
    const auto &preflight = forward_context.mamba_metadata;
    if (!preflight.input_offsets || !preflight.init_state_indices || !preflight.final_state_indices || !preflight.input_offsets.value() || !preflight.init_state_indices.value() || !preflight.final_state_indices.value())
        throw std::invalid_argument("Kimi KDA requires explicit request offsets and state indices");
    const auto &metadata = forward_context.mamba_metadata;
    raw_trace.capture(hidden_states, "input");
    raw_trace.capture(conv_preflight, "conv_before");
    raw_trace.capture(state_preflight, "ssm_before");
    raw_trace.capture(metadata.input_offsets.value(), "offsets");
    raw_trace.capture(metadata.init_state_indices.value(), "read_rows");
    raw_trace.capture(metadata.final_state_indices.value(), "write_rows");
    raw_trace.capture(A_log_.tensor(), "a_log");
    raw_trace.capture(dt_bias_.tensor(), "dt_bias");
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto input = hidden_states;
    const char* input_env_r4305=std::getenv("INFINILM_KIMI_KDA_INPUT_M1_R4305");
    const bool input_enabled_r4305=input_env_r4305&&std::string(input_env_r4305)=="1";
    if(input_env_r4305&&std::string(input_env_r4305)!="0"&&std::string(input_env_r4305)!="1")
        throw std::invalid_argument("4305 input projection flag must be 0 or 1");
    auto flag512_r8562=[](const char*name){const char*v=std::getenv(name);if(!v||std::string(v)=="0")return false;if(std::string(v)=="1")return true;throw std::invalid_argument("uniform512 flag must be0or1");};
    const bool mode512_r8562=flag512_r8562("INFINILM_KIMI_PREFILL512_R8562");
    // The serial owner driver derives ELIGIBLE from the immutable actual job:
    // kind=prefill, all actual pasts0, every request input length exactly512.
    // M divisibility alone is not eligibility (single1024/mixed1033 fallback).
    const bool eligible512_r8562=mode512_r8562&&flag512_r8562("INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562");
    const bool split512_r8562=eligible512_r8562&&batch_size==1&&seq_len>512&&seq_len<=8192&&seq_len%512==0;
    // Keep original projection outputs/views through this forward. Normal
    // provided-Runtime leases retain queued device uses; no new success fence.
    std::vector<infinilm::runtime::Tensor>held512_r8562;
    auto project512_r8562=[&](const auto&projection,infinilm::runtime::Tensor source,size_t width){
      runtime.assertOwnerThread();
      if(runtime.isGraphRecording()||!source||source->device()!=runtime.device()||source->dtype()!=infinilm::runtime::DataType::kBFloat16||(source->ndim()!=3||source->shape()[0]!=1||source->shape()[1]!=seq_len))
        throw std::invalid_argument("uniform512 requires eager sameowner BF16 packedinput");
      infinilm::runtime::Tensor output;
      try{
        output=infinilm::runtime::Tensor::empty(runtime,{1,seq_len,width},source->dtype());
        auto&counts=prefill512_counts_r8562.at(runtime.device().index());counts[0].fetch_add(1,std::memory_order_relaxed);counts[1].fetch_add(seq_len,std::memory_order_relaxed);
        for(size_t row=0;row<seq_len;row+=512){
          auto input_row=source->narrow({{1,row,512}});held512_r8562.push_back(input_row);
          auto value=projection->forward(runtime,input_row);held512_r8562.push_back(value);
          if(!value||value->shape()!=infinilm::runtime::Shape{1,512,width}||value->dtype()!=source->dtype()||value->device()!=runtime.device())throw std::logic_error("uniform512 childprojection geometry differs");
          output->narrow({{1,row,512}})->copy_from(runtime,value);counts[2].fetch_add(1,std::memory_order_relaxed);
        }
        return output;
      }catch(...){try{runtime.syncStream();}catch(...){std::terminate();}throw;}
    };
    auto selected_input_projection_r4305=[&](const auto& projection,size_t output_width){
        if(split512_r8562)return project512_r8562(projection,input,output_width);
        const bool selected=input_enabled_r4305&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&seq_len<=16;
        if(!selected)return projection->forward(runtime,input);
        runtime.assertOwnerThread();
        if(runtime.isGraphRecording()||input->dtype()!=infinilm::runtime::DataType::kBFloat16||!seq_len||batch_size!=1)
            throw std::invalid_argument("4305 bounded eager same-owner BF16 projection contract");
        auto&counts=input_projection_counts_r4305.at(runtime.device().index());
        counts[0].fetch_add(1,std::memory_order_relaxed);counts[1].fetch_add(seq_len,std::memory_order_relaxed);
        if(seq_len==1)return projection->forward(runtime,input);
        counts[2].fetch_add(1,std::memory_order_relaxed);
        auto output=infinilm::runtime::Tensor::empty(runtime,{1,seq_len,output_width},input->dtype());
        std::vector<infinilm::runtime::Tensor>held;held.reserve(seq_len);
        struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
        for(size_t i=0;i<seq_len;++i){
            auto input_row=input->narrow({{1,i,1}});held.push_back(projection->forward(runtime,input_row));auto&value=held.back();
            if(!value||value->shape()!=infinilm::runtime::Shape{1,1,output_width}||value->device()!=runtime.device()||value->dtype()!=input->dtype())
                throw std::logic_error("4305 child projection output geometry differs");
            output->narrow({{1,i,1}})->copy_from(runtime,value);
        }
        return output;
    };
    auto qkv_projected = selected_input_projection_r4305(qkv_proj_,3*local_projection_size_);
    raw_trace.capture(qkv_projected, "qkv_projected");

    const auto &conv_state = forward_context.conv_state_vec.at(layer_idx_);
    if (!qkv_projected || !conv_state || !packed_conv_weight_ || qkv_projected->ndim() != 3 || conv_state->ndim() != 3 || packed_conv_weight_->ndim() != 3 || qkv_projected->device() != forward_context.runtime->device() || conv_state->device() != forward_context.runtime->device() || packed_conv_weight_->device() != forward_context.runtime->device())
        throw std::invalid_argument("causal conv required input shape/device mismatch");
    if (packed_conv_weight_->shape()[2] != 4)
        throw std::invalid_argument("causal conv allocating API requires qualified K4");
    if (!metadata.input_offsets.value() || metadata.input_offsets.value()->ndim() != 1 || metadata.input_offsets.value()->numel() < 2)
        throw std::invalid_argument("causal conv cu_seqlens shape mismatch");
    auto qkv = infinilm::runtime::Tensor::empty(*forward_context.runtime, qkv_projected->shape(), qkv_projected->dtype());
    const std::optional<infinilm::runtime::Tensor> no_conv_tensor;
    auto conv_config = infini::ops::CausalConv1d::DefaultConfig(forward_context.runtime->device().type());
    conv_config.set_implementation_index(0);
    if (infinilm::models::kimi_native::owned_indices_active_r1213(runtime)) {
        infinilm::models::kimi_native::owned_indices_conv_r1213(runtime, qkv, conv_state, qkv_projected, packed_conv_weight_);
    } else {
        infinilm::execution::submit<infini::ops::CausalConv1d>(
            *forward_context.runtime, infinilm::execution::RecordingPolicy::HostOnly,
            conv_config,
            qkv_projected, packed_conv_weight_, conv_state, no_conv_tensor,
            std::optional<infinilm::runtime::Tensor>{metadata.input_offsets.value()}, std::optional<infinilm::runtime::Tensor>{metadata.init_state_indices.value()},
            std::optional<infinilm::runtime::Tensor>{metadata.final_state_indices.value()}, qkv, no_conv_tensor);
    }
    INFINILM_TRACE_OP(CausalConv1d, qkv);
    raw_trace.capture(qkv, "conv_output");
    auto silu_out = infinilm::runtime::Tensor::empty(*forward_context.runtime, qkv->shape(), qkv->dtype());
    INFINILM_ASSERT((*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kCpu && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kMars && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Silu>(
        (*forward_context.runtime), infini::ops::Silu::DefaultConfig((*forward_context.runtime).device().type()), qkv, silu_out);
    qkv = std::move(silu_out);
    raw_trace.capture(qkv, "silu_qkv");
    auto q = qkv->narrow({{2, 0, local_projection_size_}});
    auto k = qkv->narrow({{2, local_projection_size_, local_projection_size_}});
    auto v = qkv->narrow({{2, 2 * local_projection_size_, local_projection_size_}});
    const char* gate_env_r4576=std::getenv("INFINILM_KIMI_KDA_GATE_M1_R4571");
    int gate_policy_r4576=0;
    if(gate_env_r4576){
      std::string value(gate_env_r4576);
      if(value.size()!=1||value[0]<'0'||value[0]>'3')throw std::invalid_argument("4571 gate policy must be 0..3");
      gate_policy_r4576=value[0]-'0';
    }
    auto selected_gate_projection_r4576=[&](const auto&projection,infinilm::runtime::Tensor source,size_t output_width,int branch){
      if(split512_r8562)return project512_r8562(projection,source,output_width);
      const bool selected=gate_policy_r4576>branch&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&seq_len<=16;
      if(!selected)return projection->forward(runtime,source);
      runtime.assertOwnerThread();
      if(runtime.isGraphRecording()||!source||source->dtype()!=infinilm::runtime::DataType::kBFloat16||source->device()!=runtime.device()||!seq_len||batch_size!=1||source->ndim()!=3||source->size(0)!=1||source->size(1)!=seq_len)
        throw std::invalid_argument("4571 bounded eager same-owner BF16 gate projection contract");
      auto&counts=gate_projection_counts_r4576.at(runtime.device().index());counts[branch*3].fetch_add(1,std::memory_order_relaxed);counts[branch*3+1].fetch_add(seq_len,std::memory_order_relaxed);
      if(seq_len==1)return projection->forward(runtime,source);
      counts[branch*3+2].fetch_add(1,std::memory_order_relaxed);
      auto result=infinilm::runtime::Tensor::empty(runtime,{1,seq_len,output_width},source->dtype());
      std::vector<infinilm::runtime::Tensor>held;held.reserve(seq_len);
      struct Drain4571{infinilm::runtime::Runtime&r;~Drain4571(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
      for(size_t row=0;row<seq_len;++row){
        auto input_row=source->narrow({{1,row,1}});held.push_back(projection->forward(runtime,input_row));auto&value=held.back();
        if(!value||value->shape()!=infinilm::runtime::Shape{1,1,output_width}||value->device()!=runtime.device()||value->dtype()!=source->dtype())throw std::logic_error("4571 gate child output geometry differs");
        result->narrow({{1,row,1}})->copy_from(runtime,value);
      }
      return result;
    };
    auto f_a = selected_gate_projection_r4576(f_a_proj_,input,head_dim_,0);
    raw_trace.capture(f_a, "f_a");
    auto gate_decay_projected = selected_gate_projection_r4576(f_b_proj_,f_a,local_projection_size_,1);
    raw_trace.capture(gate_decay_projected, "gate_projected");
    auto gate_decay = gate_decay_projected->view(
        {batch_size, seq_len, local_num_heads_, head_dim_});
    auto beta_projected = selected_gate_projection_r4576(b_proj_,input,local_num_heads_,2);
    raw_trace.capture(beta_projected, "beta_projected");
    auto beta = beta_projected->view(
        {batch_size, seq_len, local_num_heads_});
    auto q_heads = q->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto k_heads = k->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto v_heads = v->view({batch_size, seq_len, local_num_heads_, head_dim_});

    auto dt_bias = dt_bias_->view({local_num_heads_, head_dim_});
    auto output = infinilm::runtime::Tensor::empty(
        *forward_context.runtime, q_heads->shape(), q_heads->dtype());
    const auto packed_shape = infinilm::runtime::Shape{1, batch_size * seq_len, local_num_heads_, head_dim_};
    auto kda_config = infini::ops::IndexedKimiDeltaAttention::DefaultConfig(forward_context.runtime->device().type());
    kda_config.set_implementation_index(0);
    // Independent default-OFF KDA-only carry boundary experiment. No projection,
    // Conv/Silu, normalization, output projection or CCL policy is changed.
    const char* serial_env_r9027=std::getenv("INFINILM_KIMI_KDA_SERIAL_R9027");
    const bool serial_enabled_r9027=serial_env_r9027&&std::string(serial_env_r9027)=="1";
    if(serial_env_r9027&&std::string(serial_env_r9027)!="0"&&std::string(serial_env_r9027)!="1")
        throw std::invalid_argument("9027 KDA serial selector must0or1");
    auto serial_kda_r9027=[&](){
        namespace rt=infinilm::runtime;
        if(!serial_enabled_r9027||seq_len<2||seq_len>8||batch_size!=1
           ||metadata.init_state_indices.value()->numel()!=1)return false;
        auto offsets=metadata.input_offsets.value(),read=metadata.init_state_indices.value(),write=metadata.final_state_indices.value();
        if(runtime.isGraphRecording()||offsets->shape()!=rt::Shape{2}||read->shape()!=rt::Shape{1}||write->shape()!=rt::Shape{1}
           ||offsets->dtype()!=rt::DataType::kInt32||read->dtype()!=rt::DataType::kInt32||write->dtype()!=rt::DataType::kInt32)
            throw std::invalid_argument("9027 eager single-request I32 metadata required");
        std::array<int32_t,2> host_offsets{},one_offsets{{0,1}};
        int32_t host_read=0,host_write=0;
        rt::Tensor step_offsets;
        std::vector<rt::Tensor> views,packed;
        struct DrainR9027{rt::Runtime&r;bool done=false;~DrainR9027()noexcept{if(!done)try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
        runtime.syncStream();
        runtime.memcpyD2H(host_offsets.data(),offsets->data(),sizeof(host_offsets));
        runtime.memcpyD2H(&host_read,read->data(),sizeof(host_read));
        runtime.memcpyD2H(&host_write,write->data(),sizeof(host_write));
        if(host_offsets[0]!=0||host_offsets[1]!=int32_t(seq_len))throw std::invalid_argument("9027 offsets must describe one complete packed request");
        if(host_read==host_write){runtime.syncStream();drain.done=true;return false;}
        if(local_num_heads_!=6||head_dim_!=128||pool!=17||output->dtype()!=rt::DataType::kBFloat16
           ||host_read<1||host_read>16||host_write<1||host_write>16
           ||std::abs(host_read-host_write)!=8
           ||infinilm::models::kimi_native::spec_math_mode_r3676(runtime)!=0
           ||infinilm::models::kimi_native::owned_indices_active_r1213(runtime))
            throw std::invalid_argument("9027 requires current H6D128 BF16 Math0 disjoint paired-state rows1..16");
        step_offsets=rt::Tensor::empty(runtime,{2},rt::DataType::kInt32);
        runtime.memcpyH2D(step_offsets->data(),one_offsets.data(),sizeof(one_offsets));
        views.reserve(seq_len*6);packed.reserve(seq_len*6);
        auto gate_heads=gate_decay->view(packed_shape);
        auto beta_heads=beta->view({1,seq_len,local_num_heads_});
        for(size_t token=0;token<seq_len;++token){
            const size_t base=views.size();
            for(const auto&t:{output,q_heads,k_heads,v_heads,gate_heads,beta_heads})views.push_back(t->narrow({{1,token,1}}));
            packed.push_back(rt::Tensor::empty(runtime,{1,1,local_num_heads_,head_dim_},output->dtype()));
            for(size_t n=1;n<6;++n)packed.push_back(views[base+n]->contiguous(runtime));
            auto token_read=token==0?read:write;
            infinilm::execution::submit<infini::ops::IndexedKimiDeltaAttention>(runtime,infinilm::execution::RecordingPolicy::HostOnly,kda_config,
                packed[base],forward_context.ssm_state_vec.at(layer_idx_),packed[base+1],packed[base+2],packed[base+3],packed[base+4],packed[base+5],
                A_log_.tensor(),dt_bias,step_offsets,token_read,write,1.0f/std::sqrt(static_cast<float>(head_dim_)),gate_lower_bound_,true);
            views[base]->copy_from(runtime,packed[base]);
            ++serial_kda_counts_r9027.at(runtime.device().index())[1];
        }
        runtime.syncStream();drain.done=true;
        auto&counts=serial_kda_counts_r9027.at(runtime.device().index());++counts[0];counts[2].fetch_add(seq_len);
        return true;
    };
    if(serial_kda_r9027()) {
        // Output is already assembled into the original full-block tensor.
    } else if (infinilm::models::kimi_native::spec_math_mode_r3676(runtime)) {
        infinilm::models::kimi_native::spec_kda_roundstep_r3676(runtime, output->view(packed_shape), forward_context.ssm_state_vec.at(layer_idx_), q_heads->view(packed_shape), k_heads->view(packed_shape), v_heads->view(packed_shape), gate_decay->view(packed_shape), beta->view({1, batch_size * seq_len, local_num_heads_}), A_log_.tensor(), dt_bias, metadata.input_offsets.value(), metadata.init_state_indices.value(), metadata.final_state_indices.value(), 1.0f/std::sqrt(static_cast<float>(head_dim_)), gate_lower_bound_, true);
    } else if (infinilm::models::kimi_native::owned_indices_active_r1213(runtime)) {
        infinilm::models::kimi_native::owned_indices_kda_r1213(runtime, output->view(packed_shape), forward_context.ssm_state_vec.at(layer_idx_), q_heads->view(packed_shape), k_heads->view(packed_shape), v_heads->view(packed_shape), gate_decay->view(packed_shape), beta->view({1, batch_size * seq_len, local_num_heads_}), A_log_.tensor(), dt_bias, 1.0f / std::sqrt(static_cast<float>(head_dim_)), gate_lower_bound_, true);
    } else {
        infinilm::execution::submit<infini::ops::IndexedKimiDeltaAttention>(
            *forward_context.runtime, infinilm::execution::RecordingPolicy::HostOnly, kda_config,
            output->view(packed_shape), forward_context.ssm_state_vec.at(layer_idx_),
            q_heads->view(packed_shape), k_heads->view(packed_shape), v_heads->view(packed_shape),
            gate_decay->view(packed_shape), beta->view({1, batch_size * seq_len, local_num_heads_}),
            A_log_.tensor(), dt_bias,
            metadata.input_offsets.value(), metadata.init_state_indices.value(), metadata.final_state_indices.value(),
            1.0f / std::sqrt(static_cast<float>(head_dim_)), gate_lower_bound_, true);
    }
    raw_trace.capture(output, "kda_output");
    raw_trace.capture(conv_state, "conv_after");
    raw_trace.capture(forward_context.ssm_state_vec.at(layer_idx_), "ssm_after");
    auto output_2d = output->view(
        {batch_size * seq_len * local_num_heads_, head_dim_});
    auto normalized_2d = o_norm_->forward(*forward_context.runtime, output_2d);
    raw_trace.capture(normalized_2d, "normalized_f32");
    auto normalized = normalized_2d->view(
        {batch_size, seq_len, local_projection_size_});
    auto output_gate_input = selected_input_projection_r4305(g_proj_,local_projection_size_);
    raw_trace.capture(output_gate_input, "output_gate_input");
    auto output_gate = infinilm::runtime::Tensor::empty((*forward_context.runtime), output_gate_input->shape(), output_gate_input->dtype());
    INFINILM_ASSERT((*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kCpu && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kMars && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Sigmoid>(
        (*forward_context.runtime), infini::ops::Sigmoid::DefaultConfig((*forward_context.runtime).device().type()), output_gate_input, output_gate);
    INFINILM_TRACE_OP(Sigmoid, output_gate);
    raw_trace.capture(output_gate, "gate_sigmoid");
    auto gated = infinilm::runtime::Tensor::empty(*forward_context.runtime, normalized->shape(), normalized->dtype());
    INFINILM_ASSERT((*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kCpu && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kMars && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Mul>(
        (*forward_context.runtime), infini::ops::Mul::DefaultConfig((*forward_context.runtime).device().type()), normalized, output_gate, gated);
    INFINILM_TRACE_OP(Mul, gated);
    raw_trace.capture(gated, "gated");
    // Diagnostic candidate: reproduce ordinary M1 RowParallelLinear math only
    // at the identified KDA output projection. Mode3 selects layer1; mode4 all
    // KDA layers. This includes the existing same-owner collective per row;
    // it deliberately avoids new broadcast-batched GEMM or unproven reduction.
    auto result = [&]() {
        const char *p9043=std::getenv("INFINILM_KIMI_ROWAR_R9043");int mode9043=0;if(p9043){if(std::string(p9043)=="1")mode9043=1;else if(std::string(p9043)=="2")mode9043=2;else if(std::string(p9043)!="0")throw std::invalid_argument("9043 mode0/1/2");}
        const char *t9043=std::getenv("INFINILM_KIMI_ROWAR_TRACE_R9043");bool trace9043=t9043&&std::string(t9043)=="1";if(t9043&&std::string(t9043)!="0"&&std::string(t9043)!="1")throw std::invalid_argument("9043 trace0/1");
        if(((mode9043&&seq_len==8)||trace9043)&&batch_size==1&&(seq_len==1||seq_len==8)){
          auto p=std::dynamic_pointer_cast<Prefill512RowR8603>(o_proj_);if(!p)throw std::logic_error("9043 original class");return p->forward_rowar_r9043(runtime,gated,mode9043);
        }
        const char *cc=std::getenv("INFINILM_KIMI_CCL512_R8603");
        if(cc && std::string(cc)!="0" && std::string(cc)!="1")
            throw std::invalid_argument("8603 flag must be0or1");
        const char *eligible=std::getenv("INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562");
        if(cc && std::string(cc)=="1" && eligible && std::string(eligible)=="1"
            && batch_size==1 && seq_len>512 && seq_len<=8192 && seq_len%512==0){
            // This object was constructed as this private derived type above.
            // Checked RTTI, never an arbitrary base-to-derived reinterpretation.
            auto selected=std::dynamic_pointer_cast<Prefill512RowR8603>(o_proj_);
            if(!selected)throw std::logic_error("8603 actual constructed Row type");
            return selected->forward_prefill512(runtime,gated);
        }
        const int mode = infinilm::models::kimi_native::spec_math_mode_r3676(runtime);
        const bool selected = mode == 4 || mode >= 6 || ((mode == 3 || mode == 5) && layer_idx_ == 1);
        if (!selected) return o_proj_->forward(runtime, gated);
        runtime.assertOwnerThread();
        if (runtime.isGraphRecording() || runtime.device().type() != infinilm::runtime::Device::Type::kHygon
            || gated->dtype() != infinilm::runtime::DataType::kBFloat16 || gated->ndim() != 3
            || gated->size(0) != 1 || gated->size(2) != local_projection_size_
            || !gated->size(1) || gated->size(1) > 128 || o_proj_->has_bias()
            || o_proj_->alpha() != 1.0F || o_proj_->output_dtype() != gated->dtype())
            throw std::invalid_argument("3932 bounded owner/BF16 KDA projection contract");
        const size_t tokens = gated->size(1);
        auto &counts = projection_counts_r3932.at(runtime.device().index());
        counts[0].fetch_add(1, std::memory_order_relaxed);
        counts[1].fetch_add(tokens, std::memory_order_relaxed);
        if (tokens == 1) return o_proj_->forward(runtime, gated);
        counts[2].fetch_add(1, std::memory_order_relaxed);
        auto shape = gated->shape(); shape.back() = o_proj_->out_features();
        auto output = infinilm::runtime::Tensor::empty(runtime, shape, gated->dtype());
        for (size_t i = 0; i < tokens; ++i) {
            auto input_row = gated->narrow({{1, i, 1}});
            auto projected_row = o_proj_->forward(runtime, input_row);
            output->narrow({{1, i, 1}})->copy_from(runtime, projected_row);
        }
        return output;
    }();
    raw_trace.capture(result, "output");
    raw_trace.finish();
    return result;
}

} // namespace infinilm::models::kimi_k3
