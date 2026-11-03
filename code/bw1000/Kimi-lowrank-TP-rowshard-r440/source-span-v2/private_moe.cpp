#include <nlohmann/json.hpp>
#include <chrono>
#include <array>
#include <string>
namespace {std::array<nlohmann::json,8> profile_r9125;thread_local std::string profile_text_r9125;}
extern "C" const char*modern_kimi_moe_profile_r9125_private_r440(int d){if(d<0||d>=8)return "{}";profile_text_r9125=profile_r9125[d].dump();return profile_text_r9125.c_str();}
#include <algorithm>
#include <cmath>
#include <cstring>
#include "kimi_n4_tile_r8171.hpp"
#include "kimi_serving_workspace_r4665.hpp"
#include "scale_exact_scope_r201.hpp"
#include <vector>
#include <tuple>
#include "kimi_spec_math_r3676.hpp"
#include <array>
#include <atomic>
#include "execution/submit.hpp"
#include "weights/weights.hpp"
#include "private_moe.hpp"
#include "projection.hpp"
// Original header supplied by private_moe.hpp
#include "weights/weights.hpp"
#include "execution/submit.hpp"

#include "global_state/global_state.hpp"
#include "runtime/runtime.hpp"

#include "engine/distributed/communication_group.hpp"
#include "engine/distributed/communication_group.hpp"
#include <exception>
#include <stdexcept>
#include <string>

extern "C" int modern_grouped_mxfp4_owned(infinilm::runtime::Runtime *, size_t, size_t, size_t, size_t, size_t, void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *);
extern "C" const char *modern_grouped_error();
extern "C" int modern_kimi_grouped_I192_r8669y(int,void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,void*,void*,void*,void*,void*,size_t,size_t,void*);
namespace {std::array<std::array<std::atomic<uint64_t>,2>,8> i192_counts_r8669y{};}
// Host-only success histogram, sampled after owning Runtime drain. No GPU read or synchronization.
namespace {std::array<std::array<std::atomic<uint64_t>,65>,8> i192_T_counts_r9129{};}
extern "C" int modern_kimi_I192_T_counts_r9129_private_r440(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=65)return 1;for(size_t t=0;t<65;++t)out[t]=i192_T_counts_r9129[device][t].load(std::memory_order_relaxed);return 0;}
namespace {
constexpr size_t compact_workspace_bytes_r8669zc = 4468480;
std::array<std::array<std::atomic<uint64_t>,2>,8> compact_workspace_counts_r8669zc{};
bool compact_workspace_optin_r8669zc() {
    const char *value = std::getenv("INFINILM_KIMI_I192_COMPACT_R8669ZC");
    return value && std::string(value) == "1";
}
}
extern "C" int modern_kimi_I192_workspace_r8669zc_private_r440(int device,uint64_t*out,size_t n) {
    if(device<0||device>=8||!out||n!=2)return 1;
    for(size_t i=0;i<2;++i)out[i]=compact_workspace_counts_r8669zc[device][i].load();
    return 0;
}

extern "C" int modern_kimi_I192_counts_r8669y_private_r440(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=2)return 1;for(size_t i=0;i<2;++i)out[i]=i192_counts_r8669y[device][i].load();return 0;}
namespace {
std::array<std::atomic<uint64_t>,8> i192_caller_success_r8669zf{};
__attribute__((noinline)) void i192_actual_caller_r8669zf(infinilm::runtime::Runtime &runtime,void*out,infinilm::runtime::Tensor &workspace,const void*input,const void*ids,const void*route_weights,const void*w13,const void*s13,const void*w2,const void*s2,size_t T=16) {
        infini::rt::DeviceGuard i192_device_guard_r8669zd{runtime.device().type(),runtime.device().index(),true};
        auto*p=reinterpret_cast<unsigned char*>(workspace->data());int code=modern_kimi_grouped_I192_r8669y(runtime.device().index(),out,p,input,ids,route_weights,w13,s13,w2,s2,p+infinilm::models::kimi_k3::serving_r4665::counts_offset,p+infinilm::models::kimi_k3::serving_r4665::routes_offset,p+infinilm::models::kimi_k3::serving_r4665::expert_offset,p+infinilm::models::kimi_k3::serving_r4665::start_offset,p+infinilm::models::kimi_k3::serving_r4665::jobs_offset,896,T,runtime.stream());
        if(code){int current=-1;const auto device_status=cudaGetDevice(&current);try{runtime.syncStream();}catch(...){std::terminate();}throw std::runtime_error("I192launchcode="+std::to_string(code)+" expected_device="+std::to_string(runtime.device().index())+" current_device="+std::to_string(current)+" cudaGetDevice_status="+std::to_string(int(device_status)));}++i192_caller_success_r8669zf.at(runtime.device().index());
}
}
namespace infinilm::models::kimi_k3 {
inline std::array<std::array<std::atomic<uint64_t>,3>,8>router_counts_r4635_private_r440{};
extern "C" int modern_kimi_router_counts_r4635_private_r440_private_r440(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=router_counts_r4635_private_r440[d][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,3>,8>route_chunk_counts_r4435_private_r440{};
extern "C" int modern_kimi_route_chunk_counts_r4435_private_r440_private_r440(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=route_chunk_counts_r4435_private_r440[d][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,6>,8>aux_counts_r4430_private_r440{};
extern "C" int modern_kimi_moe_aux_counts_r4430_private_r440_private_r440(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=6)return 1;for(size_t i=0;i<6;++i)out[i]=aux_counts_r4430_private_r440[d][i].load();return 0;}

inline std::array<std::array<std::atomic<uint64_t>,3>,8> stable_down_counts_r4012_private_r440{};
extern "C" int modern_kimi_stable_down_counts_r4012_private_r440_private_r440(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=stable_down_counts_r4012_private_r440[d][i].load();return 0;}

namespace {
thread_local bool qualify_grouped_prefill = false;
}
namespace {
thread_local bool qualify_tiled_w13_r1861 = false;
std::array<std::atomic<uint64_t>, 8> tiled_calls_r1861{};
} // namespace
extern "C" void modern_kimi_set_tiled_w13_r1861_private_r440(int enabled) { qualify_tiled_w13_r1861 = enabled != 0; }
extern "C" uint64_t modern_kimi_tiled_calls_r1861_private_r440(int device) { return device >= 0 && device < 8 ? tiled_calls_r1861[size_t(device)].load() : 0; }
extern "C" int modern_kimi_tiled_raw_r1861(int, void *, void *, const void *, const void *, const void *, const void *, const void *, const void *, const void *, size_t, void *);

extern "C" void modern_kimi_set_grouped_prefill_private_r440(int value) { qualify_grouped_prefill = value != 0; }

KimiK3MoER440::KimiK3MoER440(
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
            if(serving_r4665::allocate_workspace()) {
                // Only the exact eager TP16/I192 geometry has no partial-tail consumer.
                // Preserve all original offsets; the last reachable byte is 4468227.
                const bool compact = compact_workspace_optin_r8669zc()
                    && rank_info.tp_size == 16 && hidden_size == 7168
                    && routed_hidden_size == 3584 && expert_intermediate_size == 3072
                    && model_config->get<size_t>("num_experts") == 896
                    && model_config->get<size_t>("num_experts_per_tok") == 16
                    && dtype == infinilm::runtime::DataType::kBFloat16;
                const size_t bytes = compact ? compact_workspace_bytes_r8669zc : serving_r4665::workspace_bytes;
                tiled_scratch_r1861_=infinilm::runtime::Tensor::empty(runtime,{bytes/2},infinilm::runtime::DataType::kBFloat16);
                if(compact) {
                    auto &c=compact_workspace_counts_r8669zc.at(runtime.device().index());
                    ++c[0];c[1].fetch_add(bytes);
                }
            }
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

infinilm::runtime::Tensor KimiK3MoER440::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    auto flattened = hidden_states->view({shape[0] * shape[1], shape[2]});
    const bool profile=std::getenv("INFINILM_KIMI_PROFILE_R9125")&&std::string(std::getenv("INFINILM_KIMI_PROFILE_R9125"))=="1";
    nlohmann::json prof;auto tick=std::chrono::steady_clock::now();auto mark=[&](const char*name){if(profile){runtime.syncStream();auto now=std::chrono::steady_clock::now();prof["drained_segment_ms"][name]=std::chrono::duration<double,std::milli>(now-tick).count();tick=now;}};

    auto routing_pair_r4635=[&](){
      const char*flag=std::getenv("INFINILM_KIMI_ROUTER_M1_R4635");
      if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("4635 router flag0or1");
      const bool selected=flag&&std::string(flag)=="1"&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&flattened->size(0)<=16;
      if(!selected)return gate_->forward(runtime,flattened);
      runtime.assertOwnerThread();
      if(runtime.isGraphRecording()||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||tp_size_!=8||shape.size()!=3||shape[0]!=1||flattened->ndim()!=2||!flattened->size(0)||flattened->size(1)!=7168||flattened->dtype()!=infinilm::runtime::DataType::kBFloat16||flattened->device()!=runtime.device())throw std::invalid_argument("4635 sameownerTP8eagerBF16routergeometry");
      const size_t T=flattened->size(0);auto&counts=router_counts_r4635_private_r440.at(runtime.device().index());counts[0].fetch_add(1);counts[1].fetch_add(T);
      if(T==1)return gate_->forward(runtime,flattened);
      counts[2].fetch_add(1);auto rw=infinilm::runtime::Tensor::empty(runtime,{T,16},infinilm::runtime::DataType::kFloat32);auto ri=infinilm::runtime::Tensor::empty(runtime,{T,16},infinilm::runtime::DataType::kInt32);
      std::vector<infinilm::runtime::Tensor>held;held.reserve(2*T);
      struct Drain4635{infinilm::runtime::Runtime&r;~Drain4635(){try{r.syncStream();}catch(...){std::terminate();}}}drain{runtime};
      for(size_t i=0;i<T;++i){auto row=flattened->narrow({{0,i,1}});auto [w,k]=gate_->forward(runtime,row);
        if(!w||!k||w->shape()!=infinilm::runtime::Shape{1,16}||k->shape()!=infinilm::runtime::Shape{1,16}||w->dtype()!=infinilm::runtime::DataType::kFloat32||k->dtype()!=infinilm::runtime::DataType::kInt32||w->device()!=runtime.device()||k->device()!=runtime.device())throw std::logic_error("4635 routerchildgeometry");
        held.push_back(w);held.push_back(k);rw->narrow({{0,i,1}})->copy_from(runtime,w);ri->narrow({{0,i,1}})->copy_from(runtime,k);
      }
      return std::make_tuple(rw,ri);
    }();auto [routing_weights,selected_experts]=routing_pair_r4635;mark("router");
    auto routed_input = [&]() {
        const int mode = infinilm::models::kimi_native::spec_math_mode_r3676(runtime);
        const bool selected = mode >= 6 || (mode == 5 && stable_down_layer_r4012_ == 1);
        if (!selected) return kimi_r440::project_scoped(runtime, routed_expert_down_proj_, flattened, 0,tp_size_,communicator_);
        runtime.assertOwnerThread();
        if (runtime.isGraphRecording() || runtime.device().type() != infinilm::runtime::Device::Type::kHygon
            || flattened->ndim() != 2 || flattened->dtype() != infinilm::runtime::DataType::kBFloat16
            || !flattened->size(0) || flattened->size(0) > 128)
            throw std::invalid_argument("4012 bounded nativeBF16 replicated projection contract");
        auto &count = stable_down_counts_r4012_private_r440.at(runtime.device().index());
        count[0].fetch_add(1,std::memory_order_relaxed);count[1].fetch_add(flattened->size(0),std::memory_order_relaxed);
        if (flattened->size(0) == 1) return kimi_r440::project_scoped(runtime, routed_expert_down_proj_, flattened, 0,tp_size_,communicator_);
        count[2].fetch_add(1,std::memory_order_relaxed);
        auto shape = flattened->shape();shape.back() = routed_expert_down_proj_->out_features();
        auto output = infinilm::runtime::Tensor::empty(runtime,shape,flattened->dtype());
        for(size_t i=0;i<flattened->size(0);++i){
            auto input_row=flattened->narrow({{0,i,1}});
            auto projected=kimi_r440::project_scoped(runtime, routed_expert_down_proj_, input_row, 0,tp_size_,communicator_);
            output->narrow({{0,i,1}})->copy_from(runtime,projected);
        }
        return output;
    }();
    mark("replicated_down_projection");
    const auto &weights = experts_->mxfp4_weights();
    auto routed = infinilm::runtime::Tensor::empty(runtime, routed_input->shape(), routed_input->dtype());
    const int serving_policy_r4665=serving_r4665::policy();
    const char*i192_flag_r8669y=std::getenv("INFINILM_KIMI_I192_GROUPED_R8669Y");
    const bool i192_on_r8669y=i192_flag_r8669y&&std::string(i192_flag_r8669y)=="1";
    if(i192_on_r8669y && tp_size_==16 && serving_policy_r4665==1 && infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==0 && runtime.device().type()==infinilm::runtime::Device::Type::kHygon && routed_input->ndim()==2 && routed_input->size(1)==3584 && (routed_input->size(0)==16 || (std::getenv("INFINILM_KIMI_BATCH_I192_R9125") && std::string(std::getenv("INFINILM_KIMI_BATCH_I192_R9125"))=="1" && routed_input->size(0)>=8 && routed_input->size(0)<=64)) && routed_input->dtype()==infinilm::runtime::DataType::kBFloat16 && routed_input->is_contiguous() && selected_experts->shape()==infinilm::runtime::Shape{routed_input->size(0),16} && weights.packed_w13->shape()==infinilm::runtime::Shape{896,384,1792} && weights.w13_scale->shape()==infinilm::runtime::Shape{896,384,112} && weights.packed_w2->shape()==infinilm::runtime::Shape{896,3584,96} && weights.w2_scale->shape()==infinilm::runtime::Shape{896,3584,6}) {
        runtime.assertOwnerThread();if(runtime.isGraphRecording()||!tiled_scratch_r1861_||(tiled_scratch_r1861_->nbytes()!=serving_r4665::workspace_bytes && !(compact_workspace_optin_r8669zc() && flattened->size(1)==7168 && tiled_scratch_r1861_->nbytes()==compact_workspace_bytes_r8669zc))||tiled_scratch_r1861_->device()!=runtime.device()||tiled_scratch_r1861_->dtype()!=infinilm::runtime::DataType::kBFloat16||!tiled_scratch_r1861_->is_contiguous()||selected_experts->dtype()!=infinilm::runtime::DataType::kInt32||routing_weights->dtype()!=infinilm::runtime::DataType::kFloat32)throw std::logic_error("I192sameownedworkspace");
        i192_actual_caller_r8669zf(runtime,routed->data(),tiled_scratch_r1861_,routed_input->data(),selected_experts->data(),routing_weights->data(),weights.packed_w13->data(),weights.w13_scale->data(),weights.packed_w2->data(),weights.w2_scale->data(),routed_input->size(0));auto&c=i192_counts_r8669y.at(runtime.device().index());++c[0];c[1].fetch_add(routed_input->size(0));i192_T_counts_r9129.at(runtime.device().index()).at(routed_input->size(0)).fetch_add(1,std::memory_order_relaxed);
    } else if(serving_policy_r4665 && runtime.device().type()==infinilm::runtime::Device::Type::kHygon && routed_input->size(0)>=8 && routed_input->size(0)<=serving_r4665::capacity && routed_input->dtype()==infinilm::runtime::DataType::kBFloat16 && routed_input->size(1)==3584 && weights.packed_w13->size(0)==896 && weights.packed_w2->size(2)*2==384 && selected_experts->size(1)==16){
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
    mark("expert_W13_W2");
    if (tp_size_ > 1 && communicator_ != nullptr) {
        const char*flag=std::getenv("INFINILM_KIMI_MOE_AUX_M1_R4430");
        const bool selected=flag&&std::string(flag)=="1"&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&routed->size(0)<=16;
        if(selected){
            if(runtime.isGraphRecording()||tp_size_!=8||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||routed->ndim()!=2||routed->size(1)!=3584||!routed->size(0)||routed->dtype()!=infinilm::runtime::DataType::kBFloat16)throw std::invalid_argument("4435 eagerTP8BF16routeH3584chunk8contract");
            const size_t T=routed->size(0);auto&count=route_chunk_counts_r4435_private_r440.at(runtime.device().index());count[0].fetch_add(1);count[1].fetch_add(T);count[2].fetch_add((T+7)/8);
            for(size_t i=0;i<T;i+=8){auto row=routed->narrow({{0,i,T-i<8?T-i:8}});infinilm::engine::distributed::allreduce_(runtime,row,row,infinicclSum,communicator_);}
        }else infinilm::engine::distributed::allreduce_(runtime,routed,routed,infinicclSum,communicator_);
    }
    mark("routed_allreduce");
    routed = routed_expert_norm_->forward(runtime, routed);mark("routed_norm");
    const char*aux_flag_r4430=std::getenv("INFINILM_KIMI_MOE_AUX_M1_R4430");
    if(aux_flag_r4430&&std::string(aux_flag_r4430)!="0"&&std::string(aux_flag_r4430)!="1")throw std::invalid_argument("4430 auxflag0or1");
    const bool aux_selected_r4430=aux_flag_r4430&&std::string(aux_flag_r4430)=="1"&&infinilm::models::kimi_native::spec_math_mode_r3676(runtime)==11&&flattened->size(0)<=16;
    if(aux_selected_r4430&&(runtime.isGraphRecording()||tp_size_!=8||runtime.device().type()!=infinilm::runtime::Device::Type::kHygon||shape.size()!=3||shape[0]!=1||!flattened->size(0)||flattened->dtype()!=infinilm::runtime::DataType::kBFloat16))throw std::invalid_argument("4430 boundedTP8 eagerBF16 auxM1contract");
    if(aux_selected_r4430){auto&count=aux_counts_r4430_private_r440.at(runtime.device().index());count[3].fetch_add(1);count[4].fetch_add(flattened->size(0));if(flattened->size(0)>1)count[5].fetch_add(1);}
    if(aux_selected_r4430&&flattened->size(0)>1){auto output=infinilm::runtime::Tensor::empty(runtime,{flattened->size(0),shape[2]},routed->dtype());for(size_t i=0;i<flattened->size(0);++i){auto row=routed->narrow({{0,i,1}});auto z=kimi_r440::project_scoped(runtime, routed_expert_up_proj_, row, 1,tp_size_,communicator_);output->narrow({{0,i,1}})->copy_from(runtime,z);}routed=output->view(shape);}
    else routed=kimi_r440::project_scoped(runtime, routed_expert_up_proj_, routed, 1,tp_size_,communicator_)->view(shape);
    if(aux_selected_r4430){auto&count=aux_counts_r4430_private_r440.at(runtime.device().index());count[0].fetch_add(1);count[1].fetch_add(flattened->size(0));if(flattened->size(0)>1)count[2].fetch_add(1);}
    mark("replicated_up_projection");
    auto shared=[&](){
        if(!aux_selected_r4430||flattened->size(0)==1)return shared_experts_->forward(runtime,hidden_states);
        auto output=infinilm::runtime::Tensor::empty(runtime,shape,hidden_states->dtype());auto flatout=output->view({flattened->size(0),shape[2]});
        for(size_t i=0;i<flattened->size(0);++i){auto row=flattened->narrow({{0,i,1}})->view({1,1,shape[2]});auto z=shared_experts_->forward(runtime,row);flatout->narrow({{0,i,1}})->copy_from(runtime,z->view({1,shape[2]}));}return output;
    }();
    mark("shared_compute_including_its_TP_reduce");
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, routed->shape(), routed->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), routed, shared, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);mark("add");
    if(profile){std::vector<int32_t> ids(selected_experts->numel());runtime.memcpyD2H(ids.data(),selected_experts->data(),ids.size()*4);std::vector<int>counts(896);for(auto e:ids){if(e<0||e>=896)throw std::runtime_error("9125 route out of range");++counts[e];}size_t U=0,mx=0;for(auto n:counts){U+=n>0;mx=std::max(mx,size_t(n));}prof["T"]=flattened->size(0);prof["route_counts"]=counts;prof["unique_experts"]=U;prof["max_expert_rows"]=mx;prof["route_count"]=ids.size();prof["routed_allreduce_tensor_bytes"]=flattened->size(0)*3584*2;prof["shared_allreduce_tensor_bytes"]=flattened->size(0)*7168*2;prof["weight_bytes_unique_expert_lowerbound_per_rank"]=U*(384*1792+384*112+3584*96+3584*6);prof["profile_perturbs_stream_overlap"]=true;prof["not_HBM_measured_bytes"]=true;profile_r9125[runtime.device().index()]=prof;}
    return ops_output;
}

} // namespace infinilm::models::kimi_k3
