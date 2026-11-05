#include "deep_chain.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using Chain = std::array<decode_ccl::ChainPoint,107>;

struct DeepChainOps {
    C& context;
    std::array<void*,107> buffers{};
    std::array<std::vector<unsigned char>,107> host_inputs{};
    std::vector<unsigned char> host_output;
    Clock::time_point start, enqueued;
    double enqueue_ms{}, wall_ms{};
    float event_ms{};

    explicit DeepChainOps(C& owner) : context(owner) {}

    void allocate(const Chain& points) {
        for (std::size_t i = 0; i < points.size(); ++i) {
            const auto& p = points[i];
            host_inputs[i] = decode_ccl::chain_values(p, i, context.rank, false);
            ck(cudaMalloc(&buffers[i], p.bytes));
        }
    }

    void release() noexcept {
        try {
            // Never return memory to the allocator before this owner's work drains.
            fence();
            for (auto& buffer : buffers) {
                if (buffer) ck(cudaFree(buffer));
                buffer = nullptr;
            }
        } catch (...) {
            std::terminate();
        }
    }

    void seed(std::size_t slot, const decode_ccl::ChainPoint& p) {
        ck(cudaMemcpyAsync(buffers[slot], host_inputs[slot].data(), p.bytes,
                           cudaMemcpyHostToDevice, context.stream));
    }

    void fence() { ck(cudaStreamSynchronize(context.stream)); }

    void begin_timing() {
        ck(cudaEventRecord(context.a, context.stream));
        start = Clock::now();
    }

    void allreduce_in_place(std::size_t slot, const decode_ccl::ChainPoint& p) {
        const auto dtype = p.dtype == 9 ? infinicclBFloat16 : infinicclFloat32;
        cc(infinicclAllReduce(buffers[slot], buffers[slot], p.bytes / decode_ccl::width(p.dtype),
                             dtype, infinicclSum, context.comm, static_cast<void*>(context.stream)));
    }

    void end_enqueue() {
        enqueued = Clock::now();
        ck(cudaEventRecord(context.b, context.stream));
    }

    void end_timing() {
        const auto end = Clock::now();
        enqueue_ms = std::chrono::duration<double,std::milli>(enqueued - start).count();
        wall_ms = std::chrono::duration<double,std::milli>(end - start).count();
        ck(cudaEventElapsedTime(&event_ms, context.a, context.b));
    }

    void validate(std::size_t slot, const decode_ccl::ChainPoint& p) {
        host_output.resize(p.bytes);
        ck(cudaMemcpyAsync(host_output.data(), buffers[slot], p.bytes,
                           cudaMemcpyDeviceToHost, context.stream));
        fence();
        if (host_output != decode_ccl::chain_values(p, slot, context.rank, true))
            throw std::runtime_error("Deep collective chain output mismatch");
    }
};
}

extern "C" const char* decode_ccl_ar_chain(void* raw, std::size_t past, int fence_each, int samples) {
    auto* c = static_cast<C*>(raw);
    try {
        if (!c || c->owner != std::this_thread::get_id() || c->ranks != 8 ||
            (fence_each != 0 && fence_each != 1) || samples < 3 || samples > 40)
            throw std::invalid_argument("Deep chain owner, TP, fence mode or sample count");
        const auto points = decode_ccl::deep_chain(past);
        infini::rt::DeviceGuard guard{D::Type::kHygon, c->device, true};
        DeepChainOps ops(*c);
        try {
            ops.allocate(points);
            std::size_t bytes = 0;
            for (const auto& point : points) bytes += point.bytes;
            nlohmann::json q = {
                {"TP", c->ranks}, {"rank", c->rank}, {"past", past},
                {"inplace", true}, {"allreduce_calls_per_iteration", points.size()},
                {"API_bytes_per_iteration_per_rank", bytes},
                {"timed_stream_fences", fence_each ? points.size() : 1},
                {"samples", nlohmann::json::array()}
            };
            for (int i = 0; i < samples + 3; ++i) {
                decode_ccl::chain_once(ops, points, fence_each != 0);
                q["samples"].push_back({
                    {"warmup", i < 3}, {"stream_elapsed_ms", ops.event_ms},
                    {"enqueue_phase_wall_ms", ops.enqueue_ms}, {"wall_ms", ops.wall_ms}
                });
            }
            q["every_slot_output_checked"] = true;
            q["seed_and_validation_outside_timing"] = true;
            result = q.dump();
        } catch (...) {
            ops.release();
            throw;
        }
        ops.release();
        return result.c_str();
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}
