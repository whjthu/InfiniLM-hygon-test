#pragma once

// Private standalone full-logits component. No Model integration or default change.
#include "layers/linear/linear.hpp"
#include "engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <limits>
#include <stdexcept>

namespace infinilm::layers::linear {
class VocabParallelR8296 final : public ColumnParallelLinear {
    static infinilm::runtime::Runtime &checked_runtime(
        infinilm::runtime::Runtime &runtime, size_t hidden, size_t vocab,
        const engine::distributed::RankInfo &rank) {
        runtime.assertOwnerThread();
        if (runtime.isGraphRecording()) {
            throw std::logic_error("r8296 eager-only head construction");
        }
        if (!hidden || !vocab || (rank.tp_size != 1 && rank.tp_size != 8
                && rank.tp_size != 16 && rank.tp_size != 32)
            || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size
            || vocab % static_cast<size_t>(rank.tp_size) != 0
            || rank.device != runtime.device() || (rank.tp_size > 1 && !rank.comm)
            || rank.pp_size <= 0 || rank.pp_stage < 0 || rank.pp_stage >= rank.pp_size
            || rank.tp_size > std::numeric_limits<int>::max() / rank.pp_size
            || rank.world_size != rank.tp_size * rank.pp_size
            || rank.world_rank != rank.pp_stage * rank.tp_size + rank.tp_rank) {
            throw std::invalid_argument("r8296 invalid vocab/head/rank metadata");
        }
        return runtime;
    }
public:
    VocabParallelR8296(infinilm::runtime::Runtime &runtime, size_t hidden, size_t vocab,
                      infinilm::runtime::DataType dtype,
                      std::optional<infinilm::runtime::DataType> output_dtype,
                      const engine::distributed::RankInfo &rank)
        : ColumnParallelLinear(checked_runtime(runtime, hidden, vocab, rank), hidden, vocab,
              std::make_shared<infinilm::quantization::NoneQuantization>(nullptr, output_dtype),
              false, dtype, rank.tp_rank, rank.tp_size),
          owner_(&runtime), hidden_(hidden), vocab_(vocab), parts_(rank.tp_size),
          comm_(rank.comm) {}

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      infinilm::runtime::Tensor &input) const {
        runtime.assertOwnerThread();
        if (&runtime != owner_ || runtime.device() != device_ || !input
            || input->device() != device_ || input->ndim() < 2
            || input->shape().back() != hidden_ || input->dtype() != dtype_) {
            throw std::invalid_argument("r8296 input/Runtime mismatch");
        }
        if (runtime.isGraphRecording()
            || runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
            throw std::logic_error("r8296 eager-only full-logits head");
        }
        const auto shape = input->shape();
        size_t rows = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) {
            if (!shape[i] || rows > std::numeric_limits<size_t>::max() / shape[i]) {
                throw std::invalid_argument("r8296 empty/overflow input rows");
            }
            rows *= shape[i];
        }
        if (rows > std::numeric_limits<size_t>::max() / vocab_) {
            throw std::invalid_argument("r8296 full-logits element overflow");
        }
        // All backing retained through success fence or fatal-on-drain-failure unwind.
        // This review implementation adds a fence; no speed claim or graph support.
        infinilm::runtime::Tensor local, gathered, result;
        try {
            local = ColumnParallelLinear::forward(runtime, input);
            if (parts_ == 1) {
                result = local;
            } else {
                const size_t width = vocab_ / parts_;
                gathered = infinilm::runtime::Tensor::empty(runtime, {parts_ * rows, width}, local->dtype());
                engine::distributed::allgather_(runtime, gathered, local, comm_);
                auto output_shape = shape;
                output_shape.back() = vocab_;
                result = gathered->view({parts_, rows, width})->permute({1, 0, 2})
                    ->contiguous(runtime)->view(output_shape);
            }
            runtime.syncStream();
            return result;
        } catch (...) {
            const auto failure = std::current_exception();
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
            std::rethrow_exception(failure);
        }
    }
private:
    infinilm::runtime::Runtime *const owner_;
    const size_t hidden_, vocab_, parts_;
    const infinicclComm_t comm_; // borrowed; Engine must outlive drained model
};
} // namespace infinilm::layers::linear
