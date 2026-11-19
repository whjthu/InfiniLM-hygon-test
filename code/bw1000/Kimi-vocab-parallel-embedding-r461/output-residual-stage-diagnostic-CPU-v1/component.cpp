#include "output_residual_tiles.hpp"
#include "stage.hpp"
#include "execution/submit.hpp"
#include "layers/linear/linear.hpp"
#include "layers/normalization/rmsnorm.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"
#include <base/gemm.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace rt = infinilm::runtime;
namespace shape = kimi_output_residual_tile64;
namespace {
using Clock = std::chrono::steady_clock;
using BF16 = std::uint16_t;
constexpr std::size_t value_words = shape::full_tokens * shape::residuals * shape::hidden;
constexpr std::size_t output_words = shape::full_tokens * shape::hidden;
constexpr std::size_t guard_words = 128;
constexpr BF16 canary = 0x3f25;
constexpr std::size_t minimum_free_bytes = std::size_t{4} << 30;
thread_local std::string last_error;

struct ResidualProbe {
    std::shared_ptr<rt::Runtime> runtime;
    std::unique_ptr<infinilm::layers::RMSNorm> output_attn_res_norm_;
    std::unique_ptr<infinilm::layers::linear::ReplicatedLinear> output_attn_res_proj_;
    rt::Tensor values, backing, input_host, input_readback, norm_host, projection_host, output, output_host;
    std::vector<BF16> host_values, observed_values;
    ~ResidualProbe() {
        if (runtime) {
            try {
                kimi_stage::mark("runtime_drain.enter");
                runtime->syncStream();
                kimi_stage::mark("runtime_drain.return");
            } catch (...) {
                std::terminate();
            }
        }
    }
    rt::Tensor apply(rt::Runtime &, const rt::Tensor &, const rt::Tensor &, size_t, bool) const;
};

#include "residual_method.inc"

void cuda_ok(cudaError_t rc) {
    if (rc != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(rc));
    }
}

void sample_driver_memory(std::uint64_t *point) {
    std::size_t available = 0, total = 0;
    kimi_stage::mark("cuda_memory.enter");
    cuda_ok(cudaMemGetInfo(&available, &total));
    kimi_stage::mark("cuda_memory.return");
    point[0] = available;
    point[1] = total;
    if (available < minimum_free_bytes || available > total) {
        throw std::runtime_error("Kimi residual component 4 GiB driver floor");
    }
}

rt::Tensor host_data(BF16 *ptr, const rt::Shape &dims) {
    return rt::Tensor::from_blob(ptr, dims, rt::DataType::kBFloat16, {rt::Device::Type::kCpu, 0});
}
} // namespace

extern "C" const char *kimi_output_residual_error_r461() {
    return last_error.c_str();
}

// One arm per dedicated process. Metrics are dispatch through synchronized
// completion, D2H and input audit, and the full operation. Memory is sampled
// before setup, after setup, and after synchronized dispatch with output live.
extern "C" int kimi_output_residual_component_r461(
    int mode, int pre_transpose, double epsilon,
    BF16 *values, std::size_t nv, BF16 *norm, std::size_t nn,
    BF16 *projection, std::size_t np, BF16 *output, std::size_t no,
    double *metrics, std::size_t nm, std::uint64_t *memory, std::size_t ns) {
    const auto start = Clock::now();
    ResidualProbe probe;
    try {
        if ((mode != 0 && mode != 1) || (pre_transpose != 0 && pre_transpose != 1)
            || !std::isfinite(epsilon) || epsilon <= 0
            || !values || nv != value_words || !norm || nn != shape::hidden
            || !projection || np != shape::hidden || !output || no != output_words
            || !metrics || nm != 3 || !memory || ns != 6) {
            throw std::invalid_argument("invalid Kimi residual component inputs");
        }
        int device_count = -1;
        kimi_stage::mark("cuda_device_count.enter");
        cuda_ok(cudaGetDeviceCount(&device_count));
        kimi_stage::mark("cuda_device_count.return");
        if (device_count != 1) {
            throw std::invalid_argument("Kimi residual component requires one visible device");
        }
        kimi_stage::mark("cuda_set_device.enter");
        cuda_ok(cudaSetDevice(0));
        kimi_stage::mark("cuda_set_device.return");
        sample_driver_memory(memory);
        kimi_stage::mark("runtime_create.enter");
        probe.runtime = rt::Runtime::create({rt::Device::Type::kHygon, 0});
        kimi_stage::mark("runtime_create.return");
        auto &runtime = *probe.runtime;
        kimi_stage::mark("owner_thread_setup.enter");
        runtime.enableOwnerDrainFailStop();
        runtime.assertOwnerThread();
        kimi_stage::mark("owner_thread_setup.return");
        const auto type = rt::DataType::kBFloat16;
        kimi_stage::mark("norm_create.enter");
        probe.output_attn_res_norm_ =
            std::make_unique<infinilm::layers::RMSNorm>(runtime, shape::hidden, epsilon, type);
        kimi_stage::mark("norm_create.return");
        kimi_stage::mark("linear_create.enter");
        probe.output_attn_res_proj_ =
            std::make_unique<infinilm::layers::linear::ReplicatedLinear>(runtime, shape::hidden, 1, false, type);
        kimi_stage::mark("linear_create.return");
        probe.norm_host = host_data(norm, {shape::hidden});
        probe.projection_host = host_data(projection, {1, shape::hidden});
        kimi_stage::mark("weight_load.enter");
        probe.output_attn_res_norm_->load_parameter(runtime, "weight", probe.norm_host);
        probe.output_attn_res_proj_->load_parameter(runtime, "weight", probe.projection_host);
        kimi_stage::mark("weight_load.return");
        kimi_stage::mark("weight_postload.enter");
        probe.output_attn_res_norm_->process_weights_after_loading(runtime, pre_transpose != 0);
        probe.output_attn_res_proj_->process_weights_after_loading(runtime, pre_transpose != 0);
        kimi_stage::mark("weight_postload.return");
        probe.host_values.assign(value_words + 2 * guard_words, canary);
        std::memcpy(probe.host_values.data() + guard_words, values, nv * sizeof(BF16));
        probe.input_host = host_data(probe.host_values.data(), {probe.host_values.size()});
        kimi_stage::mark("residual_device_allocate.enter");
        probe.backing = rt::Tensor::empty(runtime, {probe.host_values.size()}, type);
        kimi_stage::mark("residual_device_allocate.return");
        probe.values = probe.backing->narrow({{0, guard_words, value_words}})
            ->view({shape::full_tokens, shape::residuals, shape::hidden});
        kimi_stage::mark("residual_h2d_enqueue.enter");
        probe.backing->copy_from(runtime, probe.input_host);
        kimi_stage::mark("residual_h2d_enqueue.return");
        // The method reads only the hidden shape. Reuse a contiguous alias of
        // the backing storage to supply that shape without another allocation.
        const auto hidden = probe.backing->narrow({{0, guard_words, output_words}})
            ->view({1, shape::full_tokens, shape::hidden});
        kimi_stage::mark("preforward_sync.enter");
        runtime.syncStream();
        sample_driver_memory(memory + 2);
        kimi_stage::mark("preforward_sync.return");
        const auto dispatch = Clock::now();
        kimi_stage::mark("forward.enter");
        probe.output = probe.apply(runtime, hidden, probe.values, shape::residuals - 1, mode == 1);
        kimi_stage::mark("forward.return");
        kimi_stage::mark("forward_sync.enter");
        runtime.syncStream();
        const auto done = Clock::now();
        kimi_stage::mark("forward_sync.return");
        sample_driver_memory(memory + 4);
        if (probe.output->shape() != rt::Shape{1, shape::full_tokens, shape::hidden}
            || probe.output->dtype() != type || !probe.output->is_contiguous()) {
            throw std::runtime_error("Kimi residual output layout differs");
        }
        probe.output_host = host_data(output, {1, shape::full_tokens, shape::hidden});
        probe.observed_values.resize(probe.host_values.size());
        probe.input_readback = host_data(probe.observed_values.data(), {probe.observed_values.size()});
        kimi_stage::mark("d2h_enqueue_and_sync.enter");
        probe.output_host->copy_from(runtime, probe.output);
        probe.input_readback->copy_from(runtime, probe.backing);
        runtime.syncStream();
        kimi_stage::mark("d2h_enqueue_and_sync.return");
        if (probe.observed_values != probe.host_values) {
            throw std::runtime_error("Kimi residual input storage or canary changed");
        }
        metrics[0] = std::chrono::duration<double>(done - dispatch).count();
        metrics[1] = std::chrono::duration<double>(Clock::now() - done).count();
        metrics[2] = std::chrono::duration<double>(Clock::now() - start).count();
        return 0;
    } catch (const std::exception &e) {
        last_error = e.what();
    } catch (...) {
        last_error = "non-standard exception in Kimi residual component";
    }
    return 1;
}
