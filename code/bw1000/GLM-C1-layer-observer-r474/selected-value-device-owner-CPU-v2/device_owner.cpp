#include "geometry.hpp"
#include "glm_moe_dsa_attention_prepare.hpp"
#include "glm_rope_bound_r4279.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace rt = infinilm::runtime;
namespace model = infinilm::models::glm_moe_dsa;
namespace dist = infinilm::engine::distributed;
namespace global = glm_global_tp16_r470;
namespace glm_nov_value::tp16 { struct SelectedPairBinding; }
using Pair = glm_nov_value::tp16::SelectedPairBinding;
extern "C" int glm_tp16_selected_pair_create(model::AttentionPrepare*, rt::Runtime*,
                                              std::size_t, Pair**) noexcept;
extern "C" int glm_tp16_selected_pair_measure(model::AttentionPrepare*, rt::Runtime*,
    const Pair*, const rt::Tensor*, const rt::Tensor*, const rt::Tensor*, int,
    std::uint64_t*, std::size_t) noexcept;
extern "C" int glm_tp16_selected_pair_destroy(rt::Runtime*, Pair*) noexcept;

namespace {
constexpr std::size_t capacity = 32768;
constexpr std::size_t floor_bytes = std::size_t{8} << 30;
thread_local std::string error;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void memory_floor() {
    std::size_t free = 0, total = 0;
    require(cudaMemGetInfo(&free, &total) == cudaSuccess && free >= floor_bytes,
            "8 GiB free memory floor");
}

struct Context {
    std::shared_ptr<rt::Runtime> runtime;
    std::unique_ptr<model::AttentionPrepare> prepare;
    Pair* pair = nullptr;
    infinicclComm_t world = nullptr, tp = nullptr;
    int rank = -1;
    bool failed = false;

    ~Context() noexcept {
        if (!runtime) return;
        try {
            runtime->assertOwnerThread();
            runtime->syncStream();
            if (pair) require(glm_tp16_selected_pair_destroy(runtime.get(), pair) == 0,
                              "selected-value binding destroy");
            if (prepare) require(modern_glm_fp8_runtime_clear_r809(runtime.get()) == 0,
                                 "FP8 Runtime cache cleanup");
            prepare.reset();
            if (tp) require(infinicclCommDestroy(tp) == infinicclSuccess, "TP destroy");
            if (world) require(infinicclCommDestroy(world) == infinicclSuccess, "world destroy");
            runtime->trimMemory();
            std::weak_ptr<rt::Runtime> weak = runtime;
            runtime.reset();
            require(weak.expired(), "Runtime retained");
        } catch (...) { std::terminate(); }
    }

    rt::Runtime& owner() const {
        runtime->assertOwnerThread();
        int device = -1;
        require(cudaGetDevice(&device) == cudaSuccess && device == rank % 8 &&
                !runtime->isGraphRecording(), "wrong owner device or graph mode");
        return *runtime;
    }
};

struct Drain {
    rt::Runtime& runtime;
    ~Drain() noexcept {
        try { runtime.syncStream(); }
        catch (...) { std::terminate(); }
    }
};

void copy_weight(Context& ctx, const char* name, const rt::Shape& shape, rt::DataType dtype,
                 const void* bytes, std::size_t size) {
    auto& r = ctx.owner();
    auto tensor = ctx.prepare->state_dict().at(name).tensor();
    require(bytes && tensor && tensor->device() == r.device() && tensor->is_contiguous() &&
            tensor->shape() == shape && tensor->dtype() == dtype &&
            tensor->nbytes() == size, "checkpoint tensor shape/dtype/size");
    Drain drain{r};
    r.memcpyH2D(tensor->data(), bytes, size, false);
}
}

extern "C" const char* glm_selected_device_error() { return error.c_str(); }

// All 16 owner processes must pass the same two real CCL IDs. The owner
// supplies private, checkpoint-verified raw rank bytes to load() below.
extern "C" Context* glm_selected_device_create(int node, int device, const char* config,
    const void* world_id, const void* tp_id, std::size_t uid_size) {
    try {
        require(config && world_id && tp_id && uid_size == sizeof(infinicclUniqueId) &&
                std::memcmp(world_id, tp_id, uid_size) != 0, "distinct exact CCL IDs");
        const auto cfg = global::J::parse(config);
        const auto placement = global::placement(node, device);
        global::config(cfg);
        global::Scope scope(cfg);
        int visible = 0;
        require(cudaGetDeviceCount(&visible) == cudaSuccess && visible == 8 &&
                cudaSetDevice(device) == cudaSuccess, "eight-device node and owner index");
        memory_floor();
        auto ctx = std::make_unique<Context>();
        ctx->rank = placement.global;
        ctx->runtime = rt::Runtime::create({rt::Device::Type::kHygon, device});
        ctx->runtime->enableOwnerDrainFailStop();
        auto& r = ctx->owner();
        infini::rt::DeviceGuard guard{r.device().type(), device, true};
        infinicclUniqueId world{}, tp{};
        std::memcpy(&world, world_id, uid_size);
        std::memcpy(&tp, tp_id, uid_size);
        require(infinicclCommInitRank(&ctx->world, 16, world, ctx->rank) == infinicclSuccess,
                "world16 CCL init");
        require(infinicclCommInitRank(&ctx->tp, 16, tp, ctx->rank) == infinicclSuccess,
                "TP16 CCL init");
        // The other ranks participate in both real groups. They do not allocate
        // or measure this layer because this experiment pins ranks 0 and 8.
        if (ctx->rank == 0 || ctx->rank == 8) {
            dist::RankInfo info(r.device());
            info.tp_size = 16;
            info.tp_rank = ctx->rank;
            info.comm = ctx->tp;
            info.pp_size = 1;
            info.pp_stage = 0;
            info.world_size = 16;
            info.world_rank = ctx->rank;
            info.world_comm = ctx->world;
            require(global::admitted(info), "original TP16 admission");
            model::rope_bound_r4279::Scope rope(r, true, capacity);
            ctx->prepare = std::make_unique<model::AttentionPrepare>(r, cfg, info);
        }
        r.syncStream();
        memory_floor();
        return ctx.release();
    } catch (const std::exception& exc) {
        error = exc.what();
        return nullptr;
    }
}

extern "C" int glm_selected_device_load(Context* ctx, const void* weight, std::size_t weight_bytes,
                                        const void* scale, std::size_t scale_bytes) {
    try {
        require(ctx && ctx->prepare && !ctx->failed && !ctx->pair, "one KV-B load per measured owner");
        auto& r = ctx->owner();
        copy_weight(*ctx, "kv_b_proj.weight", {1792,512}, rt::DataType::kUInt8, weight, weight_bytes);
        copy_weight(*ctx, "kv_b_proj.weight_scale_inv", {224,4}, rt::DataType::kFloat32, scale, scale_bytes);
        ctx->prepare->process_weights_after_loading(r, false);
        require(glm_tp16_selected_pair_create(ctx->prepare.get(), &r, ctx->rank, &ctx->pair) == 0,
                "native packed value bind");
        memory_floor();
        return 0;
    } catch (const std::exception& exc) {
        if (ctx) ctx->failed = true;
        error = exc.what();
        return 1;
    }
}

extern "C" int glm_selected_device_measure(Context* ctx, const std::uint16_t* latent,
    std::size_t latent_bytes, const std::uint16_t* rotary, std::size_t rotary_bytes,
    const std::int64_t* ids, std::size_t rows, std::int64_t last_visible,
    int packed_first, std::uint64_t* report, std::size_t report_size) {
    try {
        require(ctx && !ctx->failed && ctx->pair && latent && rotary && ids && report &&
                latent_bytes == capacity*512*sizeof(std::uint16_t) &&
                rotary_bytes == capacity*64*sizeof(std::uint16_t) &&
                rows >= 65 && rows <= 2048 && last_visible >= 0 &&
                last_visible < std::int64_t(capacity) && report_size == 7 &&
                (packed_first == 0 || packed_first == 1), "selected-value input inventory");
        require(ids[0] >= 0 && ids[rows-1] <= last_visible, "noncausal selected ID");
        for (std::size_t i = 1; i < rows; ++i)
            require(ids[i] > ids[i-1], "selected IDs must be sorted and unique");
        auto& r = ctx->owner();
        memory_floor();
        auto l = rt::Tensor::empty(r,{capacity,512},rt::DataType::kBFloat16);
        auto k = rt::Tensor::empty(r,{capacity,64},rt::DataType::kBFloat16);
        auto index = rt::Tensor::empty(r,{rows},rt::DataType::kInt64);
        Drain drain{r};
        r.memcpyH2D(l->data(),latent,latent_bytes,false);
        r.memcpyH2D(k->data(),rotary,rotary_bytes,false);
        r.memcpyH2D(index->data(),ids,rows*sizeof(*ids),false);
        r.syncStream();
        memory_floor();
        const auto result = glm_tp16_selected_pair_measure(ctx->prepare.get(), &r, ctx->pair,
            &l, &k, &index, packed_first, report, report_size);
        memory_floor();
        if (result == 3) {
            ctx->failed = true;
            error = "paired output mismatch or nonfinite values";
            return 3;
        }
        require(result == 0, "native selected-value call failed");
        return 0;
    } catch (const std::exception& exc) {
        if (ctx) ctx->failed = true;
        error = exc.what();
        return 1;
    }
}

extern "C" int glm_selected_device_close(Context* ctx) {
    try {
        require(ctx != nullptr, "no live selected-value owner");
        ctx->owner();
        delete ctx;
        memory_floor();
        return 0;
    } catch (const std::exception& exc) {
        error = exc.what();
        return 1;
    }
}
