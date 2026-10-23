#include "kimi_serial_mla_r4135.hpp"
#include "kimi_k3_kda_trace.hpp"
#include "kimi_spec_math_r3676.hpp"
#include "layers/linear/dense_prefill_f32_r2475.hpp"
#include "kimi_k3_mla_attention.hpp"
#include "kimi_k3_trace.hpp"
#include <cstdlib>
#include <array>
#include <set>
#include <vector>
#include <string>
#include <atomic>
#include <exception>
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"
#include "../../weights/weights.hpp"
#include "execution/submit.hpp"
#include "runtime/runtime.hpp"
#include <exception>

#include "../../global_state/global_state.hpp"

#include <cmath>
#include <stdexcept>

namespace infinilm::models::kimi_k3 {
inline std::array<std::array<std::atomic<uint64_t>,12>,8>mla_input_counts_r4465{};
extern "C" int modern_kimi_MLA_input_counts_r4465(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=12)return 1;for(size_t i=0;i<12;++i)out[i]=mla_input_counts_r4465[d][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,3>,8>mla_kvb_counts_r4222{};
extern "C" int modern_kimi_MLA_kvb_counts_r4222(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=mla_kvb_counts_r4222[device][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,3>,8>mla_output_counts_r4179{};
extern "C" int modern_kimi_MLA_output_counts_r4179(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=mla_output_counts_r4179[device][i].load();return 0;}

namespace {

// Private eager verifier control. Ordinary decode and prefill retain the parent
// path. Shape-specific grouping is diagnostic until full token/EOS validation.
template<class Projection>
infinilm::runtime::Tensor mla_input_r4465(infinilm::runtime::Runtime&runtime,
 infinilm::runtime::Tensor x,Projection&projection,size_t columns,size_t point){
 const char*flag=std::getenv("INFINILM_KIMI_MLA_INPUT_R4465");
 if(flag&&std::string(flag)!="0"&&std::string(flag)!="1"&&std::string(flag)!="2")throw std::invalid_argument("4465 MLAinput policy 0/1/2");
 const bool selected=flag&&std::string(flag)!="0"&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11;
 if(!selected)return projection->forward(runtime,x);
 runtime.assertOwnerThread();
 if(runtime.isGraphRecording()||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||runtime.device().index()<0||runtime.device().index()>=8||x->ndim()!=3||x->size(0)!=1||x->size(1)<1||x->size(1)>16||x->dtype()!=infinilm::runtime::DataType::kBFloat16||x->device()!=runtime.device()||point>=4||!columns)throw std::invalid_argument("4465 owner/eager/BF16/W<=16 MLAinput contract");
 const size_t T=x->size(1),width=point==0&&std::string(flag)=="2"?2:1;
 auto&count=mla_input_counts_r4465.at(runtime.device().index());count[3*point].fetch_add(1);count[3*point+1].fetch_add(T);count[3*point+2].fetch_add((T+width-1)/width);
 if(T<=width)return projection->forward(runtime,x);
 auto shape=x->shape();shape.back()=columns;auto result=infinilm::runtime::Tensor::empty(runtime,shape,x->dtype());std::vector<infinilm::runtime::Tensor>held;held.reserve((T+width-1)/width);
 struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
 for(size_t i=0;i<T;i+=width){const size_t n=T-i<width?T-i:width;auto row=x->narrow({{1,i,n}});held.push_back(projection->forward(runtime,row));if(held.back()->shape()!=infinilm::runtime::Shape({1,n,columns}))throw std::logic_error("4465 MLAinput outputshape");result->narrow({{1,i,n}})->copy_from(runtime,held.back());}
 return result;
}

std::array<std::atomic<uint64_t>, 8> gate_fp32_counts{}, q_prefill_counts_r1429{};
infinilm::runtime::Tensor gate_prefill_fp32(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input,
    const infinilm::runtime::Tensor &weight, bool gate_projection = true) {
    using DT = infinilm::runtime::DataType;
    if (runtime.device().type() != infinilm::runtime::Device::Type::kHygon || input->dtype() != DT::kBFloat16 || weight->dtype() != DT::kBFloat16 || weight->ndim() != 2 || input->shape().back() != weight->size(1))
        throw std::invalid_argument("MLA gate FP32 requires explicit unpacked Hygon BF16 weights");
    auto shape = input->shape();
    shape.back() = weight->size(0);
    auto rows = input->numel() / weight->size(1);
    const bool prefill_f32_r2475 = infinilm::layers::linear::use_dense_prefill_f32_r2475(runtime, input, weight, DT::kBFloat16);
    auto x = input->contiguous(runtime)->view({rows, weight->size(1)});
    auto w = weight->permute({1, 0});
    infinilm::runtime::Tensor sum;
    if (!prefill_f32_r2475) sum = infinilm::runtime::Tensor::empty(runtime, {rows, weight->size(0)}, DT::kFloat32);
    auto result = infinilm::runtime::Tensor::empty(runtime, {rows, weight->size(0)}, DT::kBFloat16);
    if (prefill_f32_r2475) {
        (void)infinilm::layers::linear::dense_prefill_f32_r2475(runtime, x, w, result, 1.0F);
    } else {
        infinilm::execution::submit<infini::ops::Gemm>(runtime, infinilm::execution::RecordingPolicy::DeviceOnly, infini::ops::Gemm::DefaultConfig(runtime.device().type()), x, w, std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1}, std::optional<float>{0}, std::optional<int>{}, std::optional<int>{}, sum);
        infinilm::execution::submit<infini::ops::Cast>(runtime, infini::ops::Cast::DefaultConfig(runtime.device().type()), sum, result);
    }
    if (gate_projection)
        gate_fp32_counts.at(runtime.device().index()).fetch_add(1, std::memory_order_relaxed);
    else
        q_prefill_counts_r1429.at(runtime.device().index()).fetch_add(1, std::memory_order_relaxed);
    return result->view(shape);
}

infinilm::runtime::Tensor build_key(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &kv_b,
    const infinilm::runtime::Tensor &k_rot,
    size_t num_heads,
    size_t qk_nope_head_dim,
    size_t qk_rope_head_dim) {
    auto key_shape = kv_b->shape();
    const size_t head_axis = key_shape.size() - 2;
    const size_t feature_dim = key_shape.size() - 1;
    key_shape[feature_dim] = qk_nope_head_dim + qk_rope_head_dim;

    auto key = infinilm::runtime::Tensor::empty(runtime, key_shape, kv_b->dtype());
    key->narrow({{feature_dim, 0, qk_nope_head_dim}})
        ->copy_from(runtime, kv_b->narrow({{feature_dim, 0, qk_nope_head_dim}}));

    auto k_rot_shape = key_shape;
    k_rot_shape[head_axis] = num_heads;
    k_rot_shape[feature_dim] = qk_rope_head_dim;
    infinilm::runtime::Strides k_rot_strides(key_shape.size());
    for (size_t i = 0; i < head_axis; ++i) {
        k_rot_strides[i] = k_rot->stride(i);
    }
    k_rot_strides[head_axis] = 0;
    k_rot_strides[feature_dim] = k_rot->stride(k_rot->ndim() - 1);
    auto k_rot_heads = k_rot->as_strided(k_rot_shape, k_rot_strides);
    key->narrow({{feature_dim, qk_nope_head_dim, qk_rope_head_dim}})
        ->copy_from(runtime, k_rot_heads);
    return key;
}

} // namespace

KimiK3MLAAttention::KimiK3MLAAttention(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("KimiK3MLAAttention construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit KimiK3MLAAttention construction resources");
    }
    if (attention_backend != backends::AttentionBackend::STATIC_ATTN && attention_backend != backends::AttentionBackend::PAGED_ATTN && attention_backend != backends::AttentionBackend::FLASH_ATTN) {
        throw std::invalid_argument("Invalid KimiK3MLAAttention attention backend");
    }
    if (attention_backend == backends::AttentionBackend::STATIC_ATTN) {
        throw std::runtime_error("KimiK3MLAAttention does not support static attention");
    }
    try {
        layer_idx_ = layer_idx;
        const size_t hidden_size = model_config->get<size_t>("hidden_size");
        q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
        kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
        qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
        qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
        q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
        v_head_dim_ = model_config->get<size_t>("v_head_dim");
        const size_t num_heads = model_config->get<size_t>("num_attention_heads");
        if (!model_config->get<bool>("mla_use_nope")) {
            throw std::runtime_error("KimiK3MLAAttention only supports K3's NoPE MLA");
        }
        if (num_heads % static_cast<size_t>(rank_info.tp_size) != 0) {
            throw std::runtime_error("KimiK3MLAAttention: num_attention_heads must be divisible by tp_size");
        }
        local_num_heads_ = num_heads / static_cast<size_t>(rank_info.tp_size);
        const auto &dtype = model_config->get_dtype();
        const double eps = model_config->get<double>("rms_norm_eps");

        q_a_proj_ = std::make_shared<std::remove_reference_t<decltype(*q_a_proj_)>>(
            runtime, hidden_size, q_lora_rank_, false, dtype);
        this->bind_child("q_a_proj", q_a_proj_);
        q_a_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*q_a_layernorm_)>>(
            runtime, q_lora_rank_, eps, dtype);
        this->bind_child("q_a_layernorm", q_a_layernorm_);
        q_b_proj_ = std::make_shared<std::remove_reference_t<decltype(*q_b_proj_)>>(
            runtime, q_lora_rank_, num_heads * q_head_dim_, false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("q_b_proj", q_b_proj_);
        kv_a_proj_with_mqa_ = std::make_shared<std::remove_reference_t<decltype(*kv_a_proj_with_mqa_)>>(
            runtime, hidden_size, kv_lora_rank_ + qk_rope_head_dim_, false, dtype);
        this->bind_child("kv_a_proj_with_mqa", kv_a_proj_with_mqa_);
        kv_a_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*kv_a_layernorm_)>>(
            runtime, kv_lora_rank_, eps, dtype);
        this->bind_child("kv_a_layernorm", kv_a_layernorm_);
        kv_b_proj_ = std::make_shared<std::remove_reference_t<decltype(*kv_b_proj_)>>(
            runtime, kv_lora_rank_, num_heads * (qk_nope_head_dim_ + v_head_dim_), false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("kv_b_proj", kv_b_proj_);
        g_proj_ = std::make_shared<std::remove_reference_t<decltype(*g_proj_)>>(
            runtime, hidden_size, num_heads * v_head_dim_, false, dtype, rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("g_proj", g_proj_);
        o_proj_ = std::make_shared<std::remove_reference_t<decltype(*o_proj_)>>(
            runtime, num_heads * v_head_dim_, hidden_size, false, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);

        softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(q_head_dim_));
        infinilm::layers::attention::init_kv_cache_quant_params(
            runtime,
            [this](const std::string &name, infinilm::weights::Weight parameter) {
                this->bind_weight(name, std::move(parameter));
            },
            kv_cache_k_scale_,
            kv_cache_v_scale_,
            model_config->get_kv_quant_scheme());
        attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
            local_num_heads_, q_head_dim_, softmax_scale_, local_num_heads_, layer_idx_,
            kv_cache_k_scale_, kv_cache_v_scale_, attention_backend,
            model_config->get_kv_quant_scheme());

    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3MLAAttention::forward(
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    KdaTraceBatch raw_trace(runtime,forward_context,layer_idx_);
    auto observe_cache_r8511 = [&](const char*phase) {
        const char*enabled=std::getenv("INFINILM_KIMI_TRACE_KV_POOL_R8511");
        const char*selected=std::getenv("INFINILM_KIMI_KDA_TRACE_LAYER");
        if(!enabled||std::string(enabled)!="1"||!selected||std::stoull(selected)!=layer_idx_)return;
        if(runtime.isGraphRecording())throw std::logic_error("KV observation requires eager owner");
        const auto&md=forward_context.attn_metadata;auto pool=forward_context.kv_cache_vec.at(layer_idx_);
        if(!pool||pool->ndim()!=5||pool->size(0)!=2||pool->size(1)==0||pool->size(2)!=256||pool->dtype()!=infinilm::runtime::DataType::kBFloat16)throw std::logic_error("KV observe actual pool layout differs");
        auto read_integer=[&](const auto&optional,const char*name){
            if(!optional||!*optional)throw std::logic_error("KV observe missing metadata");
            auto tensor=(*optional)->contiguous(runtime);std::vector<int64_t> values(tensor->numel());
            if(tensor->dtype()==infinilm::runtime::DataType::kInt32){std::vector<int32_t> v(values.size());runtime.memcpyD2H(v.data(),tensor->data(),v.size()*4);for(size_t i=0;i<v.size();++i)values[i]=v[i];}
            else if(tensor->dtype()==infinilm::runtime::DataType::kInt64)runtime.memcpyD2H(values.data(),tensor->data(),values.size()*8);
            else throw std::logic_error("KV observe metadata must remain I32/I64");
            auto point=std::string("kv_")+phase+"_"+name;raw_trace.capture(*optional,point.c_str());return values;
        };
        auto offsets=read_integer(md.input_offsets,"offsets");
        auto pasts=read_integer(md.past_sequence_lengths,"pasts");
        auto totals=read_integer(md.total_sequence_lengths,"totals");
        auto tables=read_integer(md.block_tables,"block_tables");
        auto slots=read_integer(md.slot_mapping,"slot_mapping");
        if(offsets.size()<2||pasts.size()+1!=offsets.size()||pasts.size()!=totals.size()||pasts.size()>4||(*md.block_tables)->ndim()!=2||(*md.block_tables)->size(0)!=pasts.size()||(*md.block_tables)->size(1)!=4||offsets.front()!=0||offsets.back()!=int64_t(slots.size()))throw std::logic_error("KV observe actual request metadata differs");
        std::set<size_t> active;
        for(size_t i=0;i<pasts.size();++i){
            if(pasts[i]<0||totals[i]<pasts[i]||totals[i]>1024||offsets[i+1]<=offsets[i]||totals[i]-pasts[i]!=offsets[i+1]-offsets[i])throw std::logic_error("KV observe actual lengths differ");
            size_t blocks=(size_t(totals[i])+pool->size(2)-1)/pool->size(2);
            if(blocks>4)throw std::logic_error("KV observe page bound differs");
            for(size_t j=0;j<blocks;++j){auto id=tables[i*4+j];if(id<0||size_t(id)>=pool->size(1))throw std::logic_error("KV observe block outofrange");active.insert(size_t(id));}
            for(int64_t t=offsets[i];t<offsets[i+1];++t){auto pos=pasts[i]+t-offsets[i];auto expect=tables[i*4+size_t(pos)/pool->size(2)]*int64_t(pool->size(2))+pos%int64_t(pool->size(2));if(slots[size_t(t)]!=expect)throw std::logic_error("KV observe slot/table mismatch");}
        }
        if(active.size()>16)throw std::logic_error("KV observe activepage bound exceeded");
        for(size_t page:active){auto point=std::string("kv_")+phase+"_block_"+std::to_string(page);raw_trace.capture(pool->narrow({{1,page,1}}),point.c_str());}
    };
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    if (batch_size != 1) {
        throw std::runtime_error("KimiK3MLAAttention: paged attention expects flattened batch size 1");
    }
    auto trace = [&](const infinilm::runtime::Tensor &value, const char *point) {
        const char *directory = std::getenv("INFINILM_KIMI_TRACE_DIR");
        if (!directory || !*directory || !trace_detail_layer(layer_idx_))
            return;
        if (value->dtype() != infinilm::runtime::DataType::kBFloat16 || value->numel() % seq_len)
            throw std::runtime_error("MLA internal trace shape/dtype differs");
        raw_trace.capture(value,point);
        auto flat = value->contiguous(runtime)->view({1, seq_len, value->numel() / seq_len});
        trace_hidden(runtime, flat, forward_context, layer_idx_, point);
    };
    auto input = hidden_states;
    trace(input, "mla_input");
    // Explicit new numerical variant: isolated currentSDK621 q_a/q_b BF16
    // output GEMMs vary on actual515/517 operands. F32 accumulation/output then
    // BF16 rounding is repeat-stable in the bounded1422 replay, not yet a full
    // model correctness claim. Native decode and every other projection remain.
    const bool q_prefill_r1429 = runtime.device().type() == infinilm::runtime::Device::Type::kHygon && forward_context.attn_metadata.max_query_length > 1 && !infinilm::models::kimi_native::spec_decode_projection_r3676(runtime);
    auto q_lora = q_prefill_r1429 ? gate_prefill_fp32(runtime, input, q_a_proj_->weight(), false) : mla_input_r4465(runtime,input,q_a_proj_,q_lora_rank_,0);
    trace(q_lora, "mla_q_a");
    q_lora = q_a_layernorm_->forward(runtime, q_lora);
    trace(q_lora, "mla_q_norm");
    auto q = q_prefill_r1429 ? gate_prefill_fp32(runtime, q_lora, q_b_proj_->weight(), false) : mla_input_r4465(runtime,q_lora,q_b_proj_,local_num_heads_*q_head_dim_,1);
    trace(q, "mla_q_b");
    auto compressed = mla_input_r4465(runtime,input,kv_a_proj_with_mqa_,kv_lora_rank_+qk_rope_head_dim_,2);
    trace(compressed, "mla_kv_a");
    auto compressed_kv = compressed->narrow({{2, 0, kv_lora_rank_}});
    auto k_rot = compressed->narrow({{2, kv_lora_rank_, qk_rope_head_dim_}});
    auto normalized_kv = kv_a_layernorm_->forward(runtime, compressed_kv);
    trace(normalized_kv, "mla_kv_norm");

    auto kv = [&]() {
        if(infinilm::models::kimi_native::spec_math_mode_r3676(runtime)!=11)
            return kv_b_proj_->forward(runtime,normalized_kv);
        runtime.assertOwnerThread();
        if(runtime.isGraphRecording()||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||runtime.device().index()<0||runtime.device().index()>=8
          ||normalized_kv->ndim()!=3||normalized_kv->size(0)!=1||normalized_kv->size(1)<1||normalized_kv->size(1)>128||normalized_kv->size(2)!=kv_lora_rank_
          ||normalized_kv->dtype()!=infinilm::runtime::DataType::kBFloat16||normalized_kv->device()!=runtime.device())
            throw std::invalid_argument("4222 MLAkvb nativeM1 owner/geometry contract");
        auto&counts=mla_kvb_counts_r4222[runtime.device().index()];counts[0].fetch_add(1);counts[1].fetch_add(normalized_kv->size(1));
        if(normalized_kv->size(1)==1)return kv_b_proj_->forward(runtime,normalized_kv);
        counts[2].fetch_add(1);auto shape=normalized_kv->shape();shape.back()=local_num_heads_*(qk_nope_head_dim_+v_head_dim_);
        auto result=infinilm::runtime::Tensor::empty(runtime,shape,normalized_kv->dtype());
        std::vector<infinilm::runtime::Tensor>held;held.reserve(normalized_kv->size(1));
        struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
        for(size_t i=0;i<normalized_kv->size(1);++i){auto row=normalized_kv->narrow({{1,i,1}});held.push_back(kv_b_proj_->forward(runtime,row));
            if(held.back()->shape()!=infinilm::runtime::Shape({1,1,shape.back()}))throw std::logic_error("4222 MLAkvb rowoutputshape");
            result->narrow({{1,i,1}})->copy_from(runtime,held.back());}
        return result;
    }();
    trace(kv, "mla_kv_b");

    q = q->view({seq_len, local_num_heads_, q_head_dim_});
    kv = kv->view({seq_len, local_num_heads_, qk_nope_head_dim_ + v_head_dim_});
    k_rot = k_rot->squeeze(0);

    const size_t feature_dim = kv->ndim() - 1;
    auto key = build_key(
        runtime, kv, k_rot, local_num_heads_, qk_nope_head_dim_, qk_rope_head_dim_);
    auto value = kv->narrow({{feature_dim, qk_nope_head_dim_, v_head_dim_}});
    // Flash/paged attention caches use the Q/K head width. Preserve the
    // model's V width by zero-padding only the unused tail before caching,
    // then take the original V slice after attention below.
    if (v_head_dim_ > q_head_dim_)
        throw std::invalid_argument("Kimi MLA V head exceeds Q/K head width");
    if (v_head_dim_ < q_head_dim_) {
        auto padded_shape = value->shape();
        padded_shape.back() = q_head_dim_;
        auto padded_value = infinilm::runtime::Tensor::empty(runtime, padded_shape, value->dtype());
        infinilm::execution::submit<infini::ops::Fill>(runtime,
                                                       infini::ops::Fill::DefaultConfig(runtime.device().type()), padded_value, double{0.0}, padded_value);
        padded_value->narrow({{feature_dim, 0, v_head_dim_}})->copy_from(runtime, value);
        value = std::move(padded_value);
    }
    trace(key, "mla_key");
    trace(value, "mla_value_padded");
    observe_cache_r8511("before");
    const int serial_mode_r4135=infinilm::models::kimi_native::spec_math_mode_r3676(runtime);
    auto padded_output = (serial_mode_r4135>=7 && forward_context.attn_metadata.max_query_length>1
        ? infinilm::models::kimi_native::serial_mla_r4135::forward(*attn_,q,key,value,forward_context,serial_mode_r4135==8)
        : attn_->forward(q,key,value,forward_context))
                             ->view({batch_size, seq_len, local_num_heads_, q_head_dim_});
    observe_cache_r8511("after");
    trace(padded_output, "mla_attention");
    auto output = infinilm::runtime::Tensor::empty(
        runtime, {batch_size, seq_len, local_num_heads_, v_head_dim_},
        padded_output->dtype());
    // Explicit candidate arithmetic: only paged prefill MLA gate GEMM emits
    // FP32 then rounds to BF16. Native decode and every other projection stay
    // on the parent path. Historical strict-reference failures remain recorded.
    auto gate_projected = runtime.device().type() == infinilm::runtime::Device::Type::kHygon && forward_context.attn_metadata.max_query_length > 1 && !infinilm::models::kimi_native::spec_decode_projection_r3676(runtime)
                              ? gate_prefill_fp32(runtime, input, g_proj_->weight())
                              : mla_input_r4465(runtime,input,g_proj_,local_num_heads_*v_head_dim_,3);
    auto gate = gate_projected->view({batch_size, seq_len, local_num_heads_, v_head_dim_});
    INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu && runtime.device().type() != infinilm::runtime::Device::Type::kMars && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
    trace(gate, "mla_gate_linear");
    infinilm::execution::submit<infini::ops::Sigmoid>(
        runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), gate, gate);
    INFINILM_TRACE_OP(Sigmoid, gate);
    trace(gate, "mla_gate_sigmoid");
    {
        const auto ops_input = padded_output->narrow({{3, 0, v_head_dim_}});
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), ops_input, gate, output);
        INFINILM_TRACE_OP(Mul, output);
    }
    trace(output, "mla_gated");
    output = output->view({batch_size, seq_len, local_num_heads_ * v_head_dim_});

    auto projected = [&]() {
        const int mode=infinilm::models::kimi_native::spec_math_mode_r3676(runtime);
        const bool selected=mode>=10||(mode==9&&layer_idx_==3);
        if(!selected)return o_proj_->forward(runtime,output);
        runtime.assertOwnerThread();
        if(runtime.isGraphRecording()||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||runtime.device().index()<0||runtime.device().index()>=8
          ||output->ndim()!=3||output->size(0)!=1||output->size(1)<1||output->size(1)>128||output->size(2)!=local_num_heads_*v_head_dim_
          ||output->dtype()!=infinilm::runtime::DataType::kBFloat16||output->device()!=runtime.device()
          ||o_proj_->has_bias()||o_proj_->alpha()!=1.f||o_proj_->output_dtype()!=output->dtype())
            throw std::invalid_argument("4179 MLAo nativeM1 owner/geometry/math contract");
        auto&counts=mla_output_counts_r4179[runtime.device().index()];counts[0].fetch_add(1);counts[1].fetch_add(output->size(1));
        if(output->size(1)==1)return o_proj_->forward(runtime,output);
        counts[2].fetch_add(1);auto shape=output->shape();shape.back()=o_proj_->out_features();auto result=infinilm::runtime::Tensor::empty(runtime,shape,output->dtype());
        std::vector<infinilm::runtime::Tensor>held;held.reserve(output->size(1));
        struct Drain{infinilm::runtime::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
        for(size_t i=0;i<output->size(1);++i){auto input_row=output->narrow({{1,i,1}});held.push_back(o_proj_->forward(runtime,input_row));result->narrow({{1,i,1}})->copy_from(runtime,held.back());}
        return result;
    }();
    trace(projected, "mla_output");
    raw_trace.finish();
    return projected;
}

} // namespace infinilm::models::kimi_k3

extern "C" uint64_t modern_kimi_mla_gate_fp32_count_r476(int device) {
    if (device < 0 || device >= 8)
        return 0;
    return infinilm::models::kimi_k3::gate_fp32_counts[size_t(device)].load(std::memory_order_relaxed);
}

extern "C" uint64_t modern_kimi_q_prefill_fp32_calls_r1429(int device) {
    if (device < 0 || device >= 8)
        return 0;
    return infinilm::models::kimi_k3::q_prefill_counts_r1429[size_t(device)].load(std::memory_order_relaxed);
}
