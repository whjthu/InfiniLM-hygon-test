#pragma once
// Fixed, private eight-owner embedding probe. No callable/function-pointer API.
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
namespace kimi_embedding_probe_r461 {
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool released=false,cancelled=false;
    std::size_t ready=0;
};
struct Request {
    const void *worker=nullptr,*context=nullptr;
    std::uint64_t epoch=0;
    std::size_t device=0;
    std::vector<std::int64_t> ids;
    std::shared_ptr<Gate> gate;
    std::vector<std::uint16_t> output;
    bool started=false,completed=false;
};
struct Slots {
    std::mutex mutex;
    std::array<std::shared_ptr<Request>,8> requests{};
};
inline Slots slots;
inline void install(const std::array<std::shared_ptr<Request>,8>&requests) {
    std::lock_guard<std::mutex> lock(slots.mutex);
    for(std::size_t d=0;d<8;++d) {
        if(slots.requests[d]||!requests[d]||requests[d]->device!=d||!requests[d]->worker
            ||!requests[d]->context||!requests[d]->epoch||!requests[d]->gate)
            throw std::invalid_argument("r461 fixed owner/context/epoch admission");
        for(std::size_t previous=0;previous<d;++previous)
            if(requests[previous]->worker==requests[d]->worker)
                throw std::invalid_argument("r461 unique eight workers");
        if(d&&(requests[d]->context!=requests[0]->context||requests[d]->epoch!=requests[0]->epoch
            ||requests[d]->gate!=requests[0]->gate||requests[d]->ids!=requests[0]->ids))
            throw std::invalid_argument("r461 same immutable local request");
    }
    slots.requests=requests; // publish all eight before the first RUN
}
inline std::shared_ptr<Request> acquire(const void*worker,std::size_t device) {
    std::lock_guard<std::mutex> lock(slots.mutex);
    if(device>=8)throw std::invalid_argument("r461 device");
    std::shared_ptr<Request> request;
    for(const auto&candidate:slots.requests)
        if(candidate&&candidate->worker==worker)request=candidate;
    if(!request)return {}; // an unrelated original RUN is never intercepted
    if(request->device!=device||request->started)
        throw std::invalid_argument("r461 owner mismatch/repeated RUN");
    request->started=true;
    return request;
}
inline bool contains(const void*worker) {
    std::lock_guard<std::mutex> lock(slots.mutex);
    for(const auto&request:slots.requests)if(request&&request->worker==worker)return true;
    return false;
}
inline void release(const std::shared_ptr<Gate>&gate,bool cancel) {
    {std::lock_guard<std::mutex> lock(gate->mutex);gate->cancelled=cancel;gate->released=true;}
    gate->changed.notify_all();
}
inline void cancel_worker(const void*worker) {
    std::shared_ptr<Gate> gate;
    {
        std::lock_guard<std::mutex> lock(slots.mutex);
        for(const auto&request:slots.requests)
            if(request&&request->worker==worker)gate=request->gate;
    }
    if(gate)release(gate,true);
}
inline void await_admission(const std::shared_ptr<Request>&request) {
    std::unique_lock<std::mutex> lock(request->gate->mutex);
    request->gate->changed.wait(lock,[&]{return request->gate->released;});
    if(request->gate->cancelled)throw std::runtime_error("r461 partial local submission cancelled before lookup");
}
inline void metadata_ready(const std::shared_ptr<Request>&request) {
    {std::lock_guard<std::mutex> lock(request->gate->mutex);++request->gate->ready;}
    request->gate->changed.notify_all();
}
inline bool await_all_metadata(const std::shared_ptr<Gate>&gate) {
    std::unique_lock<std::mutex> lock(gate->mutex);
    gate->changed.wait(lock,[&]{return gate->ready==8||gate->cancelled;});
    return !gate->cancelled;
}
inline void clear(const std::array<std::shared_ptr<Request>,8>&requests) {
    std::lock_guard<std::mutex> lock(slots.mutex);
    for(std::size_t d=0;d<8;++d)
        if(slots.requests[d]!=requests[d])throw std::logic_error("r461 teardown identity");
    slots.requests={}; // caller does this only after every submitted worker wait
}
}
