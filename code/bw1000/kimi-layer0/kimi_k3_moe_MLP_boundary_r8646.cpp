#include "kimi_n4_tile_r8171.hpp"
#include "kimi_serving_workspace_r4665.hpp"
#include <vector>
#include <tuple>
#include "kimi_spec_math_r3676.hpp"
#include <array>
#include <atomic>
#include "execution/submit.hpp"
#include "weights/weights.hpp"
#include "models/kimi_k3/kimi_k3_moe.hpp"
#include "kimi_k3_moe.hpp"
#include "../../weights/weights.hpp"
#include "execution/submit.hpp"

#include "global_state/global_state.hpp"
#include "runtime/runtime.hpp"

#include "engine/distributed/communication_group.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include <exception>
#include <stdexcept>
#include <string>

extern "C" int modern_grouped_mxfp4_owned(infinilm::runtime::Runtime *, size_t, size_t, size_t, size_t, size_t, void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *);
extern "C" const char *modern_grouped_error();
// Component-only/default-OFF; no member/layout/Common/SDK changes.
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <cmath>
namespace {
using MlpTensor8645=infinilm::runtime::Tensor;
bool mlp_component8645(){const char*p=std::getenv("INFINILM_KIMI_LAYER0_COMPONENT_R8645");return p&&std::string(p)=="1";}
const char*mlp_dir8645(){const char*p=std::getenv("INFINILM_KIMI_MLP_TRACE_DIR_R8645");return p&&*p?p:nullptr;}
std::array<std::atomic<uint64_t>,8>mlp_calls8645{};
void mlp_snapshot8645(infinilm::runtime::Runtime&r,const MlpTensor8645&x,const char*point,uint64_t call){
 const char*d=mlp_dir8645();if(!d)return;
 r.assertOwnerThread();
 if(!mlp_component8645()||r.isGraphRecording()||!x||x->dtype()!=infinilm::runtime::DataType::kBFloat16||x->ndim()!=3||x->size(0)!=1||!(x->size(1)==512||x->size(1)==2048)||!(x->size(2)==4224||x->size(2)==7168))throw std::invalid_argument("8645 layer0-only BF16 shape contract");
 const int rank=r.device().index();if(rank<0||rank>=8)throw std::invalid_argument("8645 rank");
 const auto shape=x->shape();const auto strides=x->strides();auto packed=x->contiguous(r);std::vector<uint16_t>raw(x->numel());r.memcpyD2H(raw.data(),packed->data(),raw.size()*2);
 size_t nonfinite=0;for(uint16_t b:raw){uint32_t bits=uint32_t(b)<<16;float f;std::memcpy(&f,&bits,4);nonfinite+=!std::isfinite(f);}
 const std::string name="mlp-rank"+std::to_string(rank)+"-call"+std::to_string(call)+"-"+point+"-bf16.bin";
 // Dedicated immutable per-arm directory is prepared by the existing controller.
 std::ofstream binary(std::string(d)+"/"+name,std::ios::binary);binary.write(reinterpret_cast<const char*>(raw.data()),raw.size()*2);binary.close();if(!binary)throw std::runtime_error("8645 raw write");
 nlohmann::json record={{"schema","mlp8645-full-real"},{"rank",rank},{"call",call},{"point",point},{"dtype","BF16"},{"shape",shape},{"strides",strides},{"packed_shape",packed->shape()},{"packed_strides",packed->strides()},{"bytes",raw.size()*2},{"file",name},{"nonfinite",nonfinite},{"logical_offsets",x->size(1)==512?std::vector<int>{0,512}:std::vector<int>{0,512,1024,1536,2048}},{"scope","real layer0 component; all actual companions, not service performance"}};
 std::ofstream journal(std::string(d)+"/mlp-rank"+std::to_string(rank)+".jsonl",std::ios::app);journal<<record.dump()<<'\n';journal.close();if(!journal)throw std::runtime_error("8645 journal write");if(nonfinite)throw std::runtime_error("8645 nonfinite");
}
bool mlp_gate5128645(){const char*p=std::getenv("INFINILM_KIMI_MLP_GATEUP512_R8645");if(p&&std::string(p)!="0"&&std::string(p)!="1")throw std::invalid_argument("8645 candidate flag");const bool yes=p&&std::string(p)=="1";if(yes&&!mlp_component8645())throw std::invalid_argument("8645 candidate restricted to layer0 component");return yes;}
}

namespace infinilm::models::kimi_k3 {
inline std::array<std::array<std::atomic<uint64_t>,3>,8>router_counts_r4635{};
extern "C" int modern_kimi_router_counts_r4635(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=router_counts_r4635[d][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,3>,8>route_chunk_counts_r4435{};
extern "C" int modern_kimi_route_chunk_counts_r4435(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=route_chunk_counts_r4435[d][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,6>,8>aux_counts_r4430{};
extern "C" int modern_kimi_moe_aux_counts_r4430(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=6)return 1;for(size_t i=0;i<6;++i)out[i]=aux_counts_r4430[d][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,3>,8> stable_down_counts_r4012{};
extern "C" int modern_kimi_stable_down_counts_r4012(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=stable_down_counts_r4012[d][i].load();return 0;}

namespace {
thread_local bool qualify_grouped_prefill = false;
}
namespace {
thread_local bool qualify_tiled_w13_r1861 = false;
std::array<std::atomic<uint64_t>, 8> tiled_calls_r1861{};
} // namespace
extern "C" void modern_kimi_set_tiled_w13_r1861(int enabled) { qualify_tiled_w13_r1861 = enabled != 0; }
extern "C" uint64_t modern_kimi_tiled_calls_r1861(int device) { return device >= 0 && device < 8 ? tiled_calls_r1861[size_t(device)].load() : 0; }
extern "C" int modern_kimi_tiled_raw_r1861(int, void *, void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *, size_t, void *);

extern "C" void modern_kimi_set_grouped_prefill(int value) { qualify_grouped_prefill = value != 0; }

std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_k3_subconfig(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                       size_t hidden_size,
                       size_t intermediate_size) {
    auto json = model_config->get_config_json();
    json["hidden_size"] = hidden_size;
    json["intermediate_size"] = intermediate_size;
    json["moe_intermediate_size"] = intermediate_size;
    return std::make_shared<infinilm::config::ModelConfig>(std::move(json));
}

KimiK3MLP::KimiK3MLP(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t intermediate_size,
    const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("KimiK3MLP construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit KimiK3MLP construction resources");
    }
    try {
        const size_t hidden_size = model_config->get<size_t>("hidden_size");
        const auto &dtype = model_config->get_dtype();
        situ_beta_ = model_config->get_or<float>("activation_situ_beta", 4.0f);
        situ_linear_beta_ = model_config->get_or<float>("activation_situ_linear_beta", 25.0f);
        auto register_fn = [this](const std::string &name, infinilm::weights::Weight parameter) {
            this->bind_weight(name, std::move(parameter));
        };
        gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
            runtime, hidden_size,
            intermediate_size,
            "gate_proj",
            "up_proj",
            register_fn,
            model_config->get_quantization_method(),
            false,
            dtype,
            rank_info);
        down_proj_ = std::make_shared<infinilm::layers::linear::RowParallelLinear>(
            runtime, intermediate_size, hidden_size, model_config->get_quantization_method(), false, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("down_proj", down_proj_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3MLP::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    auto input = hidden_states;
    const bool split = mlp_gate5128645();
    const char*directory = mlp_dir8645();
    uint64_t call=0;
    if(directory){runtime.assertOwnerThread();const int rank=runtime.device().index();if(rank<0||rank>=8)throw std::invalid_argument("8645 rank");call=mlp_calls8645.at(rank).fetch_add(1);}
    std::vector<infinilm::runtime::Tensor> held;
    struct Drain8645 {infinilm::runtime::Runtime&r;bool enabled;~Drain8645(){if(enabled)try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime,split};
    auto project=[&](){
        if(!split||input->ndim()!=3||input->size(0)!=1||input->size(1)!=2048||input->size(2)!=7168||input->dtype()!=infinilm::runtime::DataType::kBFloat16)return gate_up_proj_->forward_split(runtime,input);
        if(runtime.isGraphRecording())throw std::logic_error("8645 eager only");
        // Only candidate: change GateUp M2048 into four native M512 calls.
        // SituAndMul and full-M Down/CCL below remain byte-for-byte original.
        infinilm::runtime::Tensor combined,gate,up;
        for(size_t pos=0;pos<2048;pos+=512){
            auto part=input->narrow({{1,pos,512}});auto [g,u]=gate_up_proj_->forward_split(runtime,part);held.push_back(part);held.push_back(g);held.push_back(u);
            if(g->shape()!=infinilm::runtime::Shape{1,512,4224}||u->shape()!=g->shape()||g->dtype()!=input->dtype()||u->dtype()!=input->dtype())throw std::runtime_error("8645 actual GateUp geometry");
            if(pos==0){combined=infinilm::runtime::Tensor::empty(runtime,{1,2048,8448},g->dtype());held.push_back(combined);gate=combined->narrow({{2,0,4224}});up=combined->narrow({{2,4224,4224}});if(gate->strides().at(1)!=8448||up->strides().at(1)!=8448||gate->strides().at(2)!=1||up->strides().at(2)!=1)throw std::runtime_error("8645 fused gate/up stride");}
            gate->narrow({{1,pos,512}})->copy_from(runtime,g);up->narrow({{1,pos,512}})->copy_from(runtime,u);
        }
        return std::make_tuple(gate,up);
    };
    auto [gate, up] = project();
    mlp_snapshot8645(runtime,gate,"gate",call);mlp_snapshot8645(runtime,up,"up",call);
    auto activated = infinilm::runtime::Tensor::empty(runtime, gate->shape(), gate->dtype());
    infinilm::execution::submit<infini::ops::SituAndMul>(
        runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
        infini::ops::SituAndMul::DefaultConfig(runtime.device().type()),
        activated, gate, up, situ_beta_, situ_linear_beta_);
    mlp_snapshot8645(runtime,activated,"activated",call);
    auto output=down_proj_->forward(runtime,activated);
    mlp_snapshot8645(runtime,output,"row_reduced",call);
    return output;
}

KimiK3Experts::KimiK3Experts(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("KimiK3Experts construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit KimiK3Experts construction resources");
    }
    try {
        num_experts_ = model_config->get<size_t>("num_experts");
        hidden_size_ = model_config->get<size_t>("hidden_size");
        device_ = runtime.device();
        tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
        tp_size_ = static_cast<size_t>(rank_info.tp_size);
        const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
        if (intermediate_size % tp_size_ != 0) {
            throw std::runtime_error("KimiK3Experts: moe_intermediate_size must be divisible by tp_size");
        }
        local_intermediate_size_ = intermediate_size / tp_size_;
        register_mxfp4_experts(runtime);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
}

void KimiK3Experts::register_mxfp4_experts(infinilm::runtime::Runtime &runtime) {
    if (hidden_size_ % 32 != 0 || local_intermediate_size_ % 32 != 0) {
        throw std::runtime_error("KimiK3Experts: MXFP4 dimensions must be divisible by 32");
    }
    auto w13 = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, 2 * local_intermediate_size_, hidden_size_ / 2},
        infinilm::runtime::DataType::kUInt8);
    auto w13_scale = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, 2 * local_intermediate_size_, hidden_size_ / 32},
        infinilm::runtime::DataType::kUInt8);
    auto w2 = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 2},
        infinilm::runtime::DataType::kUInt8);
    auto w2_scale = infinilm::runtime::Tensor::empty(
        runtime, {num_experts_, hidden_size_, local_intermediate_size_ / 32},
        infinilm::runtime::DataType::kUInt8);

    auto register_packed = [&](const std::string &name,
                               const infinilm::runtime::Tensor &storage,
                               size_t expert,
                               size_t row_start,
                               size_t row_count,
                               size_t tp_dim) {
        auto expert_view = storage
                               ->narrow({{0, expert, 1}, {1, row_start, row_count}})
                               ->squeeze(0);
        this->bind_weight(
            name,
            infinilm::weights::Weight(expert_view, tp_dim, tp_rank_, tp_size_));
    };
    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        register_packed(prefix + "w1.weight_packed", w13, expert,
                        0, local_intermediate_size_, 0);
        register_packed(prefix + "w1.weight_scale", w13_scale, expert,
                        0, local_intermediate_size_, 0);
        register_packed(prefix + "w3.weight_packed", w13, expert,
                        local_intermediate_size_, local_intermediate_size_, 0);
        register_packed(prefix + "w3.weight_scale", w13_scale, expert,
                        local_intermediate_size_, local_intermediate_size_, 0);
        register_packed(prefix + "w2.weight_packed", w2, expert,
                        0, hidden_size_, 1);
        register_packed(prefix + "w2.weight_scale", w2_scale, expert,
                        0, hidden_size_, 1);
    }

    mxfp4_weights_.packed_w13 = std::move(w13);
    mxfp4_weights_.w13_scale = std::move(w13_scale);
    mxfp4_weights_.packed_w2 = std::move(w2);
    mxfp4_weights_.w2_scale = std::move(w2_scale);
}

const KimiK3Mxfp4MoeWeights &KimiK3Experts::mxfp4_weights() const {
    return mxfp4_weights_;
}

KimiK3MoE::KimiK3MoE(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("KimiK3MoE construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit KimiK3MoE construction resources");
    }
    try {
        stable_down_layer_r4012_ = layer_idx;
        (void)attention_backend;
        (void)ep_config; // Preserve the existing custom TP expert path; no new EP policy.
        const auto &dtype = model_config->get_dtype();
        const size_t hidden_size = model_config->get<size_t>("hidden_size");
        const size_t routed_hidden_size = model_config->get<size_t>("routed_expert_hidden_size");
        const size_t expert_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
        const size_t shared_intermediate_size = expert_intermediate_size * model_config->get<size_t>("num_shared_experts");
        const double eps = model_config->get<double>("rms_norm_eps");
        tp_size_ = rank_info.tp_size;
        communicator_ = rank_info.comm;

        gate_ = std::make_shared<std::remove_reference_t<decltype(*gate_)>>(
            runtime, model_config);
        this->bind_child("gate", gate_);
        routed_expert_down_proj_ = std::make_shared<std::remove_reference_t<decltype(*routed_expert_down_proj_)>>(
            runtime, hidden_size, routed_hidden_size, false, dtype);
        this->bind_child("routed_expert_down_proj", routed_expert_down_proj_);
        auto expert_config = make_kimi_k3_subconfig(
            model_config, routed_hidden_size, expert_intermediate_size);
        experts_ = std::make_shared<std::remove_reference_t<decltype(*experts_)>>(
            runtime, expert_config, rank_info);
        this->bind_child("experts", experts_);
        routed_expert_norm_ = std::make_shared<std::remove_reference_t<decltype(*routed_expert_norm_)>>(
            runtime, routed_hidden_size, eps, dtype);
        this->bind_child("routed_expert_norm", routed_expert_norm_);
        routed_expert_up_proj_ = std::make_shared<std::remove_reference_t<decltype(*routed_expert_up_proj_)>>(
            runtime, routed_hidden_size, hidden_size, false, dtype);
        this->bind_child("routed_expert_up_proj", routed_expert_up_proj_);
        shared_experts_ = std::make_shared<std::remove_reference_t<decltype(*shared_experts_)>>(
            runtime, model_config, shared_intermediate_size, rank_info);
        this->bind_child("shared_experts", shared_experts_);
        // Model-owned scratch is constructed on the Runtime owner and remains
        // alive until the model's normal drained teardown. Same-owner stream
        // ordering permits reuse across single-token forwards without malloc.
        if (runtime.device().type() == infinilm::runtime::Device::Type::kHygon) {
            if(serving_r4665::allocate_workspace())tiled_scratch_r1861_=infinilm::runtime::Tensor::empty(runtime,{serving_r4665::workspace_bytes/2},infinilm::runtime::DataType::kBFloat16);
            else tiled_scratch_r1861_=infinilm::runtime::Tensor::empty(runtime,{1,16,384},infinilm::runtime::DataType::kBFloat16);
        }

    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3MoE::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    auto flattened = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto routing_pair_r4635=[&](){
      const char*flag=std::getenv("INFINILM_KIMI_ROUTER_M1_R4635");
      if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("4635 router flag0or1");
      const bool selected=flag&&std::string(flag)=="1"&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&flattened->size(0)<=16;
      if(!selected)return gate_->forward(runtime,flattened);
      runtime.assertOwnerThread();
      if(runtime.isGraphRecording()||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||tp_size_!=8||shape.size()!=3||shape[0]!=1||flattened->ndim()!=2||!flattened->size(0)||flattened->size(1)!=7168||flattened->dtype()!=infinilm::runtime::DataType::kBFloat16||flattened->device()!=runtime.device())throw std::invalid_argument("4635 sameownerTP8eagerBF16routergeometry");
      const size_t T=flattened->size(0);auto&counts=router_counts_r4635.at(runtime.device().index());counts[0].fetch_add(1);counts[1].fetch_add(T);
      if(T==1)return gate_->forward(runtime,flattened);
      counts[2].fetch_add(1);auto rw=infinilm::runtime::Tensor::empty(runtime,{T,16},infinilm::runtime::DataType::kFloat32);auto ri=infinilm::runtime::Tensor::empty(runtime,{T,16},infinilm::runtime::DataType::kInt32);
      std::vector<infinilm::runtime::Tensor>held;held.reserve(2*T);
      struct Drain4635{infinilm::runtime::Runtime&r;~Drain4635(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
      for(size_t i=0;i<T;++i){auto row=flattened->narrow({{0,i,1}});auto [w,k]=gate_->forward(runtime,row);
        if(!w||!k||w->shape()!=infinilm::runtime::Shape{1,16}||k->shape()!=infinilm::runtime::Shape{1,16}||w->dtype()!=infinilm::runtime::DataType::kFloat32||k->dtype()!=infinilm::runtime::DataType::kInt32||w->device()!=runtime.device()||k->device()!=runtime.device())throw std::logic_error("4635 routerchildgeometry");
        held.push_back(w);held.push_back(k);rw->narrow({{0,i,1}})->copy_from(runtime,w);ri->narrow({{0,i,1}})->copy_from(runtime,k);
      }
      return std::make_tuple(rw,ri);
    }();auto [routing_weights,selected_experts]=routing_pair_r4635;
    auto routed_input = [&]() {
        const int mode = infinilm::models::kimi_native::spec_math_mode_r3676(runtime);
        const bool selected = mode >= 6 || (mode == 5 && stable_down_layer_r4012_ == 1);
        if (!selected) return routed_expert_down_proj_->forward(runtime, flattened);
        runtime.assertOwnerThread();
        if (runtime.isGraphRecording() || runtime.device().type() != infinilm::runtime::Device::Type::kHygon
            || flattened->ndim() != 2 || flattened->dtype() != infinilm::runtime::DataType::kBFloat16
            || !flattened->size(0) || flattened->size(0) > 128)
            throw std::invalid_argument("4012 bounded nativeBF16 replicated projection contract");
        auto &count = stable_down_counts_r4012.at(runtime.device().index());
        count[0].fetch_add(1,std::memory_order_relaxed);count[1].fetch_add(flattened->size(0),std::memory_order_relaxed);
        if (flattened->size(0) == 1) return routed_expert_down_proj_->forward(runtime, flattened);
        count[2].fetch_add(1,std::memory_order_relaxed);
        auto shape = flattened->shape();shape.back() = routed_expert_down_proj_->out_features();
        auto output = infinilm::runtime::Tensor::empty(runtime,shape,flattened->dtype());
        for(size_t i=0;i<flattened->size(0);++i){
            auto input_row=flattened->narrow({{0,i,1}});
            auto projected=routed_expert_down_proj_->forward(runtime,input_row);
            output->narrow({{0,i,1}})->copy_from(runtime,projected);
        }
        return output;
    }();
    const auto &weights = experts_->mxfp4_weights();
    auto routed = infinilm::runtime::Tensor::empty(runtime, routed_input->shape(), routed_input->dtype());
    const int serving_policy_r4665=serving_r4665::policy();
    if(serving_policy_r4665 && runtime.device().type()==infinilm::runtime::Device::Type::kHygon && routed_input->size(0)>=8 && routed_input->size(0)<=serving_r4665::capacity && routed_input->dtype()==infinilm::runtime::DataType::kBFloat16 && routed_input->size(1)==3584 && weights.packed_w13->size(0)==896 && weights.packed_w2->size(2)*2==384 && selected_experts->size(1)==16){
        const int variant=serving_policy_r4665==1?(routed_input->size(0)>=64?9:4):serving_policy_r4665;
        serving_r4665::launch(runtime,tiled_scratch_r1861_,routed_input->size(0),variant,routed->data(),routed_input->data(),selected_experts->data(),routing_weights->data(),weights.packed_w13->data(),weights.w13_scale->data(),weights.packed_w2->data(),weights.w2_scale->data());
    } else if (qualify_grouped_prefill && runtime.device().type() == infinilm::runtime::Device::Type::kHygon && routed_input->size(0) >= 512) {
        const auto tokens = routed_input->size(0), hidden = routed_input->size(1), experts = weights.packed_w13->size(0), intermediate = weights.packed_w2->size(2) * 2, topk = selected_experts->size(1);
        if (modern_grouped_mxfp4_owned(&runtime, tokens, experts, hidden, intermediate, topk, routed->data(), routed_input->data(), selected_experts->data(), routing_weights->data(), weights.packed_w13->data(), weights.w13_scale->data(), weights.packed_w2->data(), weights.w2_scale->data()))
            throw std::runtime_error(modern_grouped_error());
    } else if (n4_tile_r8167::enabled() && infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==0 && runtime.device().type()==infinilm::runtime::Device::Type::kHygon && routed_input->size(0)==4 && routed_input->dtype()==infinilm::runtime::DataType::kBFloat16 && routed_input->is_contiguous() && routed_input->size(1)==3584 && weights.packed_w13->size(0)==896 && weights.packed_w2->size(2)*2==384 && selected_experts->size(1)==16){
        n4_tile_r8167::launch(runtime,tiled_scratch_r1861_,routed->data(),routed_input->data(),selected_experts->data(),routing_weights->data(),weights.packed_w13->data(),weights.w13_scale->data(),weights.packed_w2->data(),weights.w2_scale->data());
    } else if (qualify_tiled_w13_r1861 && runtime.device().type() == infinilm::runtime::Device::Type::kHygon && routed_input->size(0) == 1 && routed_input->dtype() == infinilm::runtime::DataType::kBFloat16 && routed_input->size(1) == 3584 && weights.packed_w2->size(2) * 2 == 384 && selected_experts->size(1) == 16) {
        runtime.assertOwnerThread();
        infini::rt::DeviceGuard device_guard_r2024{runtime.device().type(), runtime.device().index(), true};
        if (runtime.isGraphRecording() || !tiled_scratch_r1861_)
            throw std::logic_error("1861missing eager owned scratch");
        auto code = modern_kimi_tiled_raw_r1861(runtime.device().index(), routed->data(), tiled_scratch_r1861_->data(), routed_input->data(), selected_experts->data(), routing_weights->data(), weights.packed_w13->data(), weights.w13_scale->data(), weights.packed_w2->data(), weights.w2_scale->data(), weights.packed_w13->size(0), runtime.stream());
        if (code) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
            throw std::runtime_error("1861 tiled W13 kernel launch failed code=" + std::to_string(code));
        }
        tiled_calls_r1861.at(runtime.device().index()).fetch_add(1, std::memory_order_relaxed);
    } else {
        auto moe_config = infini::ops::FusedMoeMxfp4::DefaultConfig(runtime.device().type());
        if (runtime.device().type() == infinilm::runtime::Device::Type::kHygon)
            moe_config.set_implementation_index(1); // Qualified 16-route W2 provider.
        infinilm::execution::submit<infini::ops::FusedMoeMxfp4>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, moe_config,
            routed, routed_input,
            selected_experts,
            routing_weights,
            weights.packed_w13,
            weights.w13_scale,
            weights.packed_w2,
            weights.w2_scale,
            size_t{2});
    }
    if (tp_size_ > 1 && communicator_ != nullptr) {
        const char*flag=std::getenv("INFINILM_KIMI_MOE_AUX_M1_R4430");
        const bool selected=flag&&std::string(flag)=="1"&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&routed->size(0)<=16;
        if(selected){
            if(runtime.isGraphRecording()||tp_size_!=8||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||routed->ndim()!=2||routed->size(1)!=3584||!routed->size(0)||routed->dtype()!=infinilm::runtime::DataType::kBFloat16)throw std::invalid_argument("4435 eagerTP8BF16routeH3584chunk8contract");
            const size_t T=routed->size(0);auto&count=route_chunk_counts_r4435.at(runtime.device().index());count[0].fetch_add(1);count[1].fetch_add(T);count[2].fetch_add((T+7)/8);
            for(size_t i=0;i<T;i+=8){auto row=routed->narrow({{0,i,T-i<8?T-i:8}});infinilm::engine::distributed::allreduce_(runtime,row,row,infinicclSum,communicator_);}
        }else infinilm::engine::distributed::allreduce_(runtime,routed,routed,infinicclSum,communicator_);
    }
    routed = routed_expert_norm_->forward(runtime, routed);
    const char*aux_flag_r4430=std::getenv("INFINILM_KIMI_MOE_AUX_M1_R4430");
    if(aux_flag_r4430&&std::string(aux_flag_r4430)!="0"&&std::string(aux_flag_r4430)!="1")throw std::invalid_argument("4430 auxflag0or1");
    const bool aux_selected_r4430=aux_flag_r4430&&std::string(aux_flag_r4430)=="1"&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&flattened->size(0)<=16;
    if(aux_selected_r4430&&(runtime.isGraphRecording()||tp_size_!=8||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||shape.size()!=3||shape[0]!=1||!flattened->size(0)||flattened->dtype()!=infinilm::runtime::DataType::kBFloat16))throw std::invalid_argument("4430 boundedTP8 eagerBF16 auxM1contract");
    if(aux_selected_r4430){auto&count=aux_counts_r4430.at(runtime.device().index());count[3].fetch_add(1);count[4].fetch_add(flattened->size(0));if(flattened->size(0)>1)count[5].fetch_add(1);}
    if(aux_selected_r4430&&flattened->size(0)>1){auto output=infinilm::runtime::Tensor::empty(runtime,{flattened->size(0),shape[2]},routed->dtype());for(size_t i=0;i<flattened->size(0);++i){auto row=routed->narrow({{0,i,1}});auto z=routed_expert_up_proj_->forward(runtime,row);output->narrow({{0,i,1}})->copy_from(runtime,z);}routed=output->view(shape);}
    else routed=routed_expert_up_proj_->forward(runtime,routed)->view(shape);
    if(aux_selected_r4430){auto&count=aux_counts_r4430.at(runtime.device().index());count[0].fetch_add(1);count[1].fetch_add(flattened->size(0));if(flattened->size(0)>1)count[2].fetch_add(1);}
    auto shared=[&](){
        if(!aux_selected_r4430||flattened->size(0)==1)return shared_experts_->forward(runtime,hidden_states);
        auto output=infinilm::runtime::Tensor::empty(runtime,shape,hidden_states->dtype());auto flatout=output->view({flattened->size(0),shape[2]});
        for(size_t i=0;i<flattened->size(0);++i){auto row=flattened->narrow({{0,i,1}})->view({1,1,shape[2]});auto z=shared_experts_->forward(runtime,row);flatout->narrow({{0,i,1}})->copy_from(runtime,z->view({1,shape[2]}));}return output;
    }();
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, routed->shape(), routed->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), routed, shared, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

} // namespace infinilm::models::kimi_k3
