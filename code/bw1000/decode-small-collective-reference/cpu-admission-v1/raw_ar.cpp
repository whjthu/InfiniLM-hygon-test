// Small-message decode InfiniCCL reference using the original owner and timing contract.
#include "decode_geometry.hpp"
#include <infiniccl/infiniccl.h>
#include <runtime/device.hpp>
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <thread>
namespace {thread_local std::string error,result;using D=infinilm::runtime::Device;struct C{std::thread::id owner=std::this_thread::get_id();int rank,device,ranks;bool inplace{};infinicclComm_t comm{};cudaStream_t stream{};cudaEvent_t a{},b{};void*input{};void*output{};size_t bytes{};int dtype{};std::vector<unsigned char>host;};
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void cc(infinicclResult_t x){if(x!=infinicclSuccess)throw std::runtime_error("InfiniCCL status "+std::to_string(int(x)));}
void release(C*c){if(!c)return;infini::rt::DeviceGuard guard{D::Type::kHygon,c->device,true};if(c->stream)ck(cudaStreamSynchronize(c->stream));if(c->input)ck(cudaFree(c->input));if(c->output)ck(cudaFree(c->output));if(c->a)ck(cudaEventDestroy(c->a));if(c->b)ck(cudaEventDestroy(c->b));if(c->comm)cc(infinicclCommDestroy(c->comm));if(c->stream)ck(cudaStreamDestroy(c->stream));}
}
extern "C" const char*decode_ccl_ar_error(){return error.c_str();}
extern "C" size_t decode_ccl_ar_id_size(){return sizeof(infinicclUniqueId);}
extern "C" int decode_ccl_ar_id(void*out){try{if(!out)throw std::invalid_argument("nullID");infinicclUniqueId id{};cc(infinicclGetUniqueId(&id));std::memcpy(out,&id,sizeof(id));return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int decode_ccl_ar_geometry(int ranks, size_t bytes, int dtype) {
    return decode_ccl::valid(ranks, bytes, dtype) ? 0 : 1;
}
extern "C" void*decode_ccl_ar_create(int ranks,int rank,int device,const void*idbytes){std::unique_ptr<C>c;try{if((ranks!=8&&ranks!=16)||rank<0||rank>=ranks||device<0||device>=8||rank%8!=device||!idbytes)throw std::invalid_argument("rank/device/ID");c=std::make_unique<C>();c->rank=rank;c->device=device;c->ranks=ranks;infini::rt::DeviceGuard guard{D::Type::kHygon,device,true};int actual=-1;ck(cudaGetDevice(&actual));if(actual!=device)throw std::runtime_error("guard/device mismatch");ck(cudaStreamCreate(&c->stream));ck(cudaEventCreate(&c->a));ck(cudaEventCreate(&c->b));infinicclUniqueId id{};std::memcpy(&id,idbytes,sizeof(id));cc(infinicclCommInitRank(&c->comm,ranks,id,rank));return c.release();}catch(const std::exception&e){error=e.what();if(c)try{release(c.get());}catch(...){}return nullptr;}}
extern "C" int decode_ccl_ar_prepare(void* raw, size_t bytes, int dtype, int inplace) {
    auto* c = static_cast<C*>(raw);
    try {
        if (!c || c->owner != std::this_thread::get_id() || (inplace != 0 && inplace != 1))
            throw std::invalid_argument("prepare owner or alias mode");
        auto host = decode_ccl::seed(c->ranks, c->rank, bytes, dtype);
        infini::rt::DeviceGuard guard{D::Type::kHygon, c->device, true};
        ck(cudaStreamSynchronize(c->stream));
        if (c->input) ck(cudaFree(c->input));
        c->input = nullptr;
        if (c->output) ck(cudaFree(c->output));
        c->output = nullptr;
        c->bytes = 0;
        ck(cudaMalloc(&c->input, bytes));
        if (!inplace) ck(cudaMalloc(&c->output, bytes));
        c->host = std::move(host);
        c->bytes = bytes;
        c->dtype = dtype;
        c->inplace = inplace != 0;
        return 0;
    } catch (const std::exception& e) {
        error = e.what();
        return 1;
    }
}

extern "C" const char* decode_ccl_ar_run(void* raw, int samples) {
    auto* c = static_cast<C*>(raw);
    try {
        if (!c || c->owner != std::this_thread::get_id() || !c->bytes ||
            !c->input || (!c->inplace && !c->output) || samples < 3 || samples > 40)
            throw std::invalid_argument("run owner, prepared state or samples");
        infini::rt::DeviceGuard guard{D::Type::kHygon, c->device, true};
        void* target = c->inplace ? c->input : c->output;
        const auto count = c->bytes / decode_ccl::width(c->dtype);
        const auto dtype = c->dtype == 9 ? infinicclBFloat16 : infinicclFloat32;
        nlohmann::json q = {
            {"TP", c->ranks}, {"rank", c->rank}, {"bytes_per_rank", c->bytes},
            {"dtype", c->dtype == 9 ? "BF16" : "F32"}, {"inplace", c->inplace},
            {"Model_weights", false}, {"GPU_F64", false}, {"samples", nlohmann::json::array()}
        };
        std::vector<unsigned char> host(c->bytes);
        for (int i = 0; i < samples + 3; ++i) {
            // Refill even in-place. Move reset and poisoning outside the clocks.
            ck(cudaMemcpyAsync(c->input, c->host.data(), c->bytes, cudaMemcpyHostToDevice, c->stream));
            if (!c->inplace) ck(cudaMemsetAsync(c->output, 0xff, c->bytes, c->stream));
            ck(cudaStreamSynchronize(c->stream));
            ck(cudaEventRecord(c->a, c->stream));
            const auto start = std::chrono::steady_clock::now();
            cc(infinicclAllReduce(c->input, target, count, dtype, infinicclSum, c->comm, (void*)c->stream));
            const auto enqueued = std::chrono::steady_clock::now();
            ck(cudaEventRecord(c->b, c->stream));
            ck(cudaStreamSynchronize(c->stream));
            const auto end = std::chrono::steady_clock::now();
            float ms{};
            ck(cudaEventElapsedTime(&ms, c->a, c->b));
            ck(cudaMemcpyAsync(host.data(), target, c->bytes, cudaMemcpyDeviceToHost, c->stream));
            ck(cudaStreamSynchronize(c->stream));
            if (decode_ccl::wrong_count(host, c->ranks, c->dtype))
                throw std::runtime_error("decode collective output mismatch");
            if (!c->inplace) {
                ck(cudaMemcpyAsync(host.data(), c->input, c->bytes, cudaMemcpyDeviceToHost, c->stream));
                ck(cudaStreamSynchronize(c->stream));
                if (host != c->host) throw std::runtime_error("out-of-place input changed");
            }
            q["samples"].push_back({
                {"warmup", i < 3}, {"stream_elapsed_ms", ms},
                {"host_enqueue_ms", std::chrono::duration<double,std::milli>(enqueued-start).count()},
                {"wall_ms", std::chrono::duration<double,std::milli>(end-start).count()}
            });
        }
        q["all_iterations_full_output_checked"] = true;
        q["validation_outside_timed_window"] = true;
        q["out_of_place_input_checked"] = !c->inplace;
        result = q.dump();
        return result.c_str();
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}
extern "C" int decode_ccl_ar_close(void*raw){auto*c=(C*)raw;try{if(!c||c->owner!=std::this_thread::get_id())throw std::invalid_argument("nullctx/wrongowner");release(c);delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
#include "deep-chain/deep_chain_runtime.inc"
