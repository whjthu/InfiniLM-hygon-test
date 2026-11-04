#pragma once
#include "embedding_index_r461.hpp"
#include "layers/embedding/embedding.hpp"
#include "engine/distributed/communication_group.hpp"
#include "engine/distributed/kimi_reduce_precision_r3226.hpp"
#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <array>
#include <atomic>
#include <exception>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

extern "C" int modern_kimi_masked_embedding_r461(int,void*,const void*,const void*,std::size_t,std::size_t,void*);
namespace infinilm::layers {
namespace vocab_r461 {
struct Census {
    std::atomic<bool> constructed{false},destroyed{false},loaded{false};
    std::size_t rank=0,physical_bytes=0,row_begin=0,row_end=0;
    std::uintptr_t backing=0;
    std::atomic<std::uint64_t> forwards{0},tokens{0},collectives{0},completed_fences{0};
};
inline std::array<Census,8> census{};
inline nlohmann::json snapshot() {
    nlohmann::json owners=nlohmann::json::array();
    for(std::size_t d=0;d<8;++d) {
        const auto &v=census[d];const bool built=v.constructed.load(std::memory_order_acquire);
        owners.push_back({{"device",d},{"constructed",built},{"destroyed",v.destroyed.load()},
            {"postload_checked",v.loaded.load()},{"tp_rank",v.rank},{"physical_shape",{5120,7168}},
            {"parameter_bytes",v.physical_bytes},{"global_row_begin",v.row_begin},{"global_row_end_exclusive",v.row_end},
            {"backing_address",v.backing},{"registered_weight_aliases_same_tensor",built},
            {"forward_calls",v.forwards.load()},{"tokens",v.tokens.load()},
            {"BF16_sum_collectives",v.collectives.load()},{"completed_fences",v.completed_fences.load()}});
    }
    return {{"schema",1},{"full_vocabulary",163840},{"hidden",7168},{"TP",32},
        {"kind","MaskedRowShardedBF16EmbeddingR461"},{"owners",owners},
        {"full_GPU_shadow",false},{"full_GPU_shadow_scope","source construction/loader guarantee; not allocator-pool dynamic census"}};
}
}

class VocabParallelEmbeddingR461 final : public Embedding {
    static runtime::Runtime &checked(runtime::Runtime &r,const engine::distributed::RankInfo &rank) {
        r.assertOwnerThread();
        if(r.isGraphRecording()||r.graphCaptureState()!=execution::graph::GraphManager::CaptureState::kInactive
            ||r.device().type()!=runtime::Device::Type::kHygon||r.device()!=rank.device
            ||rank.tp_size!=32||rank.pp_size!=1||rank.pp_stage!=0||rank.tp_rank<0||rank.tp_rank>=32
            ||rank.world_size!=32||rank.world_rank!=rank.tp_rank||!rank.comm)
            throw std::invalid_argument("r461 explicit eager Hygon TP32PP1 owner geometry");
        return r;
    }
public:
    VocabParallelEmbeddingR461(runtime::Runtime &r,const engine::distributed::RankInfo &rank)
        :Embedding(checked(r,rank),kimi_vocab_r461::local_rows,kimi_vocab_r461::hidden,
                   std::nullopt,runtime::DataType::kBFloat16),owner_(&r),rank_(rank.tp_rank),comm_(rank.comm),device_index_(r.device().index()) {
        // The base constructs only [5120,7168]. Rebinding changes checkpoint
        // policy on that SAME backing; no full-table Tensor is constructed.
        auto backing=weight_.tensor();
        weight_=weights::Weight(backing,0,rank_,32);
        this->bind_weight("weight",weight_);
        if(local_weights_.at("weight").tensor()->data()!=backing->data()
            ||weight_.tensor()->data()!=backing->data()||backing->shape()!=runtime::Shape{5120,7168}
            ||backing->dtype()!=runtime::DataType::kBFloat16||!backing->is_contiguous()
            ||backing->nbytes()!=kimi_vocab_r461::local_bytes)
            throw std::logic_error("r461 same backing and global checkpoint binding");
        auto &v=vocab_r461::census.at(r.device().index());
        v.rank=rank_;v.row_begin=rank_*5120;v.row_end=(rank_+1)*5120;
        v.physical_bytes=backing->nbytes();v.backing=reinterpret_cast<std::uintptr_t>(backing->data());
        v.destroyed=false;v.loaded=false;v.forwards=0;v.tokens=0;v.collectives=0;v.completed_fences=0;
        v.constructed.store(true,std::memory_order_release);
    }
    ~VocabParallelEmbeddingR461()override {
        // Original worker close drains and joins before member destruction;
        // this destructor may run on the Engine caller. No Runtime/device call.
        vocab_r461::census.at(device_index_).destroyed=true;
    }

    void process_weights_after_loading(runtime::Runtime &r,bool transpose)override {
        r.assertOwnerThread();
        if(&r!=owner_||transpose||weight_->shape()!=runtime::Shape{5120,7168}
            ||local_weights_.at("weight").tensor()->data()!=weight_->data())
            throw std::logic_error("r461 unchanged native BF16 row layout after loading");
        vocab_r461::census.at(r.device().index()).loaded=true;
    }

    runtime::Tensor forward_sharded(runtime::Runtime &r,const runtime::Tensor &ids)const {
        r.assertOwnerThread();
        const auto weights=weight_.tensor();
        if(&r!=owner_||r.isGraphRecording()||r.graphCaptureState()!=execution::graph::GraphManager::CaptureState::kInactive
            ||engine::distributed::reduce_precision_r3226::enabled(r)
            ||!ids||ids->device()!=r.device()||ids->dtype()!=runtime::DataType::kInt64
            ||ids->ndim()!=2||ids->size(0)!=1||!ids->size(1)||ids->size(1)>512
            ||weights->device()!=r.device()||weights->shape()!=runtime::Shape{5120,7168}
            ||weights->dtype()!=runtime::DataType::kBFloat16||!weights->is_contiguous())
            throw std::invalid_argument("r461 legal C1 global IDs and resident local BF16 rows");
        // Invalid GLOBAL IDs are rejected before submission by the new Native
        // CPU input boundary; the device mask only selects the owning TP rank.
        infini::rt::DeviceGuard device_guard{r.device().type(),r.device().index(),true};
        runtime::Tensor input,output;
        try {
            input=ids->contiguous(r);
            output=runtime::Tensor::empty(r,{1,ids->size(1),7168},runtime::DataType::kBFloat16);
            auto &v=vocab_r461::census.at(r.device().index());
            const int code=modern_kimi_masked_embedding_r461(r.device().index(),output->data(),input->data(),
                weights->data(),ids->size(1),rank_,r.stream());
            if(code)throw std::runtime_error("r461 masked lookup launch "+std::to_string(code));
            ++v.forwards;v.tokens.fetch_add(ids->size(1));
            // Every TP rank participates exactly once, including ranks owning
            // none of the IDs. One nonzero BF16 contributor per valid token.
            engine::distributed::allreduce_(r,output,output,infinicclSum,comm_);
            ++v.collectives;
            // Original eager collective submission does not retain temporary
            // Tensor leases to completion. Keep all leases through this fence.
            r.syncStream();++v.completed_fences;
            return output;
        }catch(...){
            auto failure=std::current_exception();
            try{r.syncStream();}catch(...){std::terminate();}
            std::rethrow_exception(failure);
        }
    }
private:
    runtime::Runtime *const owner_;
    const std::size_t rank_;
    const infinicclComm_t comm_;
    const std::size_t device_index_;
};
inline runtime::Tensor embedding_dispatch_r461(runtime::Runtime&r,const std::shared_ptr<Embedding>&embedding,const runtime::Tensor&ids) {
    if(const auto *parallel=dynamic_cast<const VocabParallelEmbeddingR461*>(embedding.get()))
        return parallel->forward_sharded(r,ids);
    return embedding->forward(r,ids);
}
}
