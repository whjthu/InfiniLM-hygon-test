#include "glm_moe_dsa_attention_prepare.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" int modern_glm_projection_host_timing_begin(void*) noexcept;
extern "C" int modern_glm_projection_host_timing_end(
    void*, std::uint64_t*, std::size_t, std::size_t*, int*) noexcept;

namespace rt = infinilm::runtime;
using infinilm::models::glm_moe_dsa::Fp8Projection;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

constexpr std::size_t floor_bytes = std::size_t{8} << 30;

std::size_t memory_floor() {
    std::size_t free = 0, total = 0;
    cuda_check(cudaMemGetInfo(&free, &total));
    require(free >= floor_bytes, "8 GiB free memory floor");
    return free;
}

struct DrainBeforeRelease {
    rt::Runtime& runtime;
    ~DrainBeforeRelease() noexcept {
        try { runtime.syncStream(); }
        catch (...) { std::terminate(); }
    }
};

template<typename T>
void set_weight(rt::Runtime& runtime, Fp8Projection& projection, const char* name,
                const std::vector<T>& data) {
    const auto target = projection.state_dict().at(name).tensor();
    require(target->is_contiguous() && target->nbytes() == data.size()*sizeof(T), "weight layout");
    auto source = rt::Tensor::empty(runtime, target->shape(), target->dtype());
    DrainBeforeRelease drain{runtime};
    runtime.memcpyH2D(source->data(), data.data(), source->nbytes(), false);
    projection.load_parameter(runtime, name, source);
}

void fill_weights(rt::Runtime& runtime, Fp8Projection& projection, std::size_t rows,
                  std::size_t columns, std::size_t global_rows) {
    std::vector<std::uint8_t> values(rows*columns);
    for (std::size_t i = 0; i < values.size(); ++i) {
        // E4M3FN values +1 and -0.5. Deliberately synthetic, never checkpoint data.
        values[i] = (i + i/columns) % 7 ? 0x38 : 0xb0;
    }
    const std::vector<float> scales(((global_rows+127)/128)*((columns+127)/128), 0.03125f);
    set_weight(runtime, projection, "weight", values);
    set_weight(runtime, projection, "weight_scale_inv", scales);
}

std::vector<std::uint16_t> copy_output(rt::Runtime& runtime, const rt::Tensor& tensor,
                                      std::size_t columns) {
    std::vector<std::uint16_t> output(columns);
    require(tensor->shape() == rt::Shape{1, columns} && tensor->is_contiguous() &&
            tensor->dtype() == rt::DataType::kBFloat16 &&
            tensor->nbytes() == output.size()*sizeof(output[0]), "output layout");
    DrainBeforeRelease drain{runtime};
    runtime.memcpyD2H(output.data(), tensor->data(), tensor->nbytes());
    runtime.syncStream();
    require(std::none_of(output.begin(), output.end(), [](auto x) { return (x&0x7f80)==0x7f80; }),
            "non-finite projection output");
    require(std::any_of(output.begin(), output.end(), [](auto x) { return (x&0x7fff)!=0; }),
            "all-zero projection output");
    return output;
}

using Outputs = std::array<std::vector<std::uint16_t>, 2>;
constexpr std::size_t row_width = 9;

struct Capture {
    rt::Runtime& runtime;
    std::array<std::uint64_t, 32*row_width> samples{};
    std::size_t count = 0;
    int dropped = 0;
    bool active = false;

    void begin() {
        require(modern_glm_projection_host_timing_begin(&runtime)==0, "timing begin");
        active = true;
    }

    void finish() {
        require(modern_glm_projection_host_timing_end(
            &runtime, samples.data(), samples.size()/row_width, &count, &dropped)==0, "timing end");
        active = false;
    }

    ~Capture() noexcept {
        if (active && modern_glm_projection_host_timing_end(
            &runtime, samples.data(), samples.size()/row_width, &count, &dropped)!=0) std::terminate();
    }
};

void exercise(rt::Runtime& runtime) {
    runtime.assertOwnerThread();
    Fp8Projection qa(runtime,2048,6144,1,0);
    // Directly owned local projection. Retain TP16's scale extent using the actual
    // partitioned Fp8Projection constructor, with rank zero and no communicator.
    Fp8Projection qb(runtime,16384,2048,16,0);
    fill_weights(runtime,qa,2048,6144,2048);
    // The state dictionary holds a local shard. Set exact shard bytes in place.
    {
        auto state = qb.state_dict();
        auto weight = state.at("weight").tensor();
        auto scale = state.at("weight_scale_inv").tensor();
        require(weight->shape()==rt::Shape{1024,2048} && weight->is_contiguous() &&
                weight->dtype()==rt::DataType::kUInt8 && weight->nbytes()==1024*2048, "local weight shape");
        require(scale->shape()==rt::Shape{128,16} && scale->is_contiguous() &&
                scale->dtype()==rt::DataType::kFloat32 && scale->nbytes()==128*16*sizeof(float),
                "replicated scale shape");
        std::vector<std::uint8_t> bytes(weight->nbytes(),0x38);
        std::vector<float> factors(128*16,0.03125f);
        DrainBeforeRelease drain{runtime};
        runtime.memcpyH2D(weight->data(),bytes.data(),bytes.size(),false);
        runtime.memcpyH2D(scale->data(),factors.data(),scale->nbytes(),false);
    }
    std::vector<std::uint16_t> inputs(6144);
    for (std::size_t i=0; i<inputs.size(); ++i) inputs[i] = i%11 ? 0x3e80 : 0xbe00;
    auto x = rt::Tensor::empty(runtime,{1,6144},rt::DataType::kBFloat16);
    {
        DrainBeforeRelease drain{runtime};
        runtime.memcpyH2D(x->data(),inputs.data(),x->nbytes(),false);
    }
    Outputs reference;
    bool have_reference = false;
    // Three same-process pairs. Record each pair independently; no perf promotion.
    const std::array<bool,6> enabled{false,true,true,false,false,true};
    for (std::size_t arm=0; arm<enabled.size(); ++arm) {
        Capture timing{runtime};
        if (enabled[arm]) timing.begin();
        auto first = qa.forward(runtime,x);
        auto second = qb.forward(runtime,first);
        if (enabled[arm]) {
            timing.finish();
            require(timing.count==2 && timing.dropped==0, "two complete timing rows");
            const auto* a = timing.samples.data();
            const auto* b = a + row_width;
            require(a[0]==2048 && a[1]==6144 && a[2]==1 && a[8]==1 &&
                    b[0]==1024 && b[1]==2048 && b[2]==1 && b[8]==0, "recorded projection geometry");
            require(a[3]==0 && b[3]==0 && a[4]==0 && b[4]==0, "expected T1 uncached inactive lease");
        }
        Outputs actual{copy_output(runtime,first,2048),copy_output(runtime,second,1024)};
        if (!have_reference) {
            reference = actual;
            have_reference = true;
        }
        require(actual==reference, "paired BF16 output mismatch");
        const auto available = memory_floor();
        std::cout << "{\"arm\":" << arm << ",\"timing\":" << (enabled[arm] ? "true" : "false")
                  << ",\"bf16_exact\":true,\"free_bytes\":" << available << ",\"samples\":[";
        for (std::size_t i=0; i<timing.count; ++i) {
            if (i) std::cout << ',';
            std::cout << '[';
            for (std::size_t j=0; j<row_width; ++j) {
                if (j) std::cout << ',';
                std::cout << timing.samples[i*row_width+j];
            }
            std::cout << ']';
        }
        std::cout << "]}" << std::endl;
    }
}

int main() {
    try {
        int devices=0;
        cuda_check(cudaGetDeviceCount(&devices));
        require(devices==1,"exactly one Slurm device");
        cuda_check(cudaSetDevice(0));
        char pci[32]{};
        cuda_check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
        memory_floor();
        auto runtime = rt::Runtime::create({rt::Device::Type::kHygon,0});
        runtime->enableOwnerDrainFailStop();
        std::weak_ptr<rt::Runtime> lifetime = runtime;
        exercise(*runtime);
        runtime->syncStream();
        runtime->trimMemory();
        runtime.reset();
        require(lifetime.expired(),"Runtime retained after close");
        memory_floor();
        std::cout << "{\"completed\":true,\"runtime_released\":true,\"pci\":\"" << pci << "\"}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
