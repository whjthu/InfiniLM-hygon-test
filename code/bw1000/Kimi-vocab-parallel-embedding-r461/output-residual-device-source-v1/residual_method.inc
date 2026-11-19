infinilm::runtime::Tensor ResidualProbe::apply(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &block_residual_storage,
    size_t block_residual_count, bool tiled) const {
    const auto shape = hidden_states->shape();
    auto values = block_residual_storage->narrow(
        {{1, 0, block_residual_count + 1}});
    namespace tiling = kimi_output_residual_tile64;
    using Tensor = infinilm::runtime::Tensor;
    using Shape = infinilm::runtime::Shape;
    using Device = infinilm::runtime::Device;
    using DType = infinilm::runtime::DataType;
    const auto *tile_setting = (tiled ? "1" : nullptr);
    Tensor normalized_full, scores;
    if (tile_setting && shape.size() == 3
        && tiling::selected(tile_setting,
            runtime.device().type() == Device::Type::kHygon
            && !runtime.isGraphRecording()
            && runtime.graphCaptureState() == infinilm::execution::graph::GraphManager::CaptureState::kInactive,
            shape[0], shape[1], shape[2], block_residual_count + 1)
        && values->shape() == Shape{tiling::full_tokens, tiling::residuals, tiling::hidden}
        && values->dtype() == DType::kBFloat16 && values->is_contiguous()) {
        scores = Tensor::empty(runtime, {tiling::full_tokens, tiling::residuals, 1}, DType::kBFloat16);
        for (std::size_t index = 0; index < tiling::tile_count; ++index) {
            const auto tile = tiling::tile(index);
            Tensor tile_values, normalized, tile_scores, target;
            try {
                tile_values = values->narrow({{0, tile.begin, tile.count}});
                if (!tile_values->is_contiguous()
                    || tile_values->shape() != Shape{tile.count, tiling::residuals, tiling::hidden}) {
                    throw std::logic_error("Kimi output residual tile input layout");
                }
                normalized = output_attn_res_norm_->forward(runtime, tile_values);
                if (!normalized || normalized->shape() != tile_values->shape()
                    || normalized->dtype() != DType::kBFloat16 || !normalized->is_contiguous()) {
                    throw std::logic_error("Kimi output residual tile norm layout");
                }
                tile_scores = output_attn_res_proj_->forward(runtime, normalized);
                if (!tile_scores || tile_scores->shape() != Shape{tile.count, tiling::residuals, 1}
                    || tile_scores->dtype() != DType::kBFloat16 || !tile_scores->is_contiguous()) {
                    throw std::logic_error("Kimi output residual tile score layout");
                }
                target = scores->narrow({{0, tile.begin, tile.count}});
                target->copy_from(runtime, tile_scores);
                // The candidate keeps all tile owners through the copy, then bounds
                // the number of live tiles before acquiring the next scratch.
                runtime.syncStream();
            } catch (...) {
                const auto error = std::current_exception();
                try {
                    runtime.syncStream();
                } catch (...) {
                    std::terminate();
                }
                std::rethrow_exception(error);
            }
        }
    } else {
        normalized_full = output_attn_res_norm_->forward(runtime, values);
        scores = output_attn_res_proj_->forward(runtime, normalized_full);
    }
    infinilm::execution::submit<infini::ops::Softmax>(
        runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), scores, std::int64_t{1}, std::optional<infini::ops::DataType>{}, scores);
    auto score_matrix = scores->view(
        {scores->size(0), scores->size(2), scores->size(1)});
    auto matmul_output_shape = score_matrix->shape();
    matmul_output_shape.back() = values->size(score_matrix->ndim() - 1);
    auto matmul_output = infinilm::runtime::Tensor::empty(runtime, matmul_output_shape, score_matrix->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), score_matrix, values,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0f},
        std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, matmul_output);
    INFINILM_TRACE_OP(Gemm, matmul_output);
    return matmul_output
        ->squeeze(1)
        ->view(shape);
}
