#include "fresh_kv_route.hpp"
#include <cassert>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rt {
using Shape = std::vector<std::size_t>;
enum class DataType { kBFloat16, f32 };
struct Runtime;
struct TensorImpl;
using Tensor = std::shared_ptr<TensorImpl>;

struct TensorImpl {
    Shape dims, strides;
    int device_id = 2;
    DataType type = DataType::kBFloat16;
    bool dense = true;
    bool pointer = true;
    uint16_t storage = 0;

    explicit TensorImpl(Shape shape) : dims(std::move(shape)), strides(dims.size(), 1) {
        for (size_t i = dims.size(); i > 1; --i) strides[i-2] = strides[i-1] * dims[i-1];
    }
    int device() const { return device_id; }
    DataType dtype() const { return type; }
    size_t ndim() const { return dims.size(); }
    size_t size(size_t i) const { return dims.at(i); }
    size_t stride(size_t i) const { return strides.at(i); }
    const Shape& shape() const { return dims; }
    bool is_contiguous() const { return dense; }
    void* data() { return pointer ? &storage : nullptr; }
    Tensor view(const Shape&);
    Tensor narrow(std::initializer_list<std::initializer_list<size_t>>) const {
        return std::make_shared<TensorImpl>(*this);
    }
    void copy_from(Runtime&, const Tensor&);
    Tensor contiguous(Runtime&) const;
    int view_fault = 0;
    Runtime* projection_owner = nullptr;
};

struct Runtime {
    bool recording = false, has_stream = true, bad_output = false;
    int project_fault = 0, next_rc = 0;
    unsigned project = 0, allocations = 0, original_copies = 0, contiguous_copies = 0;
    unsigned syncs = 0, drains = 0, marker_count = 0, launches = 0, chosen = 0;
    void* projected = nullptr;
    void* launch_projected = nullptr;
    void* launch_rotary = nullptr;
    void* launch_stream = nullptr;
    int device() const { return 2; }
    bool isGraphRecording() const { return recording; }
    void assertOwnerThread() const {}
    void* stream() { return has_stream ? this : nullptr; }
    void syncStream() { ++syncs; }
};

Tensor TensorImpl::view(const Shape& dims_) {
    auto out = std::make_shared<TensorImpl>(dims_);
    out->device_id = view_fault == 3 ? 3 : device_id;
    out->type = view_fault == 2 ? DataType::f32 : type;
    if (view_fault == 1) out->strides[0] = 1;
    if (view_fault == 4) out->dims[2] = 447;
    if (projection_owner) projection_owner->projected = out->data();
    return out;
}
void TensorImpl::copy_from(Runtime& r, const Tensor&) { ++r.original_copies; }
Tensor TensorImpl::contiguous(Runtime& r) const {
    ++r.contiguous_copies;
    return std::make_shared<TensorImpl>(*this);
}
}

namespace glm_attention_detail_v1 {
void mark(rt::Runtime& r, const char*) { ++r.marker_count; }
}

struct Work {
    rt::Runtime& r;
    std::vector<rt::Tensor> held{};
    const int entered = std::uncaught_exceptions();
    ~Work() { if (std::uncaught_exceptions() > entered) ++r.drains; }
    rt::Tensor alloc(rt::Shape shape, rt::DataType dtype) {
        auto out = std::make_shared<rt::TensorImpl>(std::move(shape));
        ++r.allocations;
        if (r.bad_output && r.allocations == 2) out->strides[0] += 1;
        out->type = dtype;
        held.push_back(out);
        return out;
    }
};

struct Projection {
    rt::Tensor forward(rt::Runtime& r, const rt::Tensor& x) const {
        ++r.project;
        auto out = std::make_shared<rt::TensorImpl>(rt::Shape{x->size(0), 1792});
        out->view_fault = r.project_fault;
        out->projection_owner = &r;
        return out;
    }
};

class AttentionPrepare {
    bool ready_ = true;
    int device_ = 2;
    size_t heads_ = 4;
    std::unique_ptr<Projection> kb_ = std::make_unique<Projection>();
public:
    std::pair<rt::Tensor,rt::Tensor> expand_cached(rt::Runtime&, const rt::Tensor&, const rt::Tensor&) const;
};

int launch_fake(unsigned arm, void* key, void* val, const void* projection, const void* rotary,
                size_t tokens, size_t heads, void* owner) {
    auto& r = *static_cast<rt::Runtime*>(owner);
    assert(key && val && key != val && projection == r.projected && rotary && tokens == 1 && heads == 4
           && r.project == 1);
    ++r.launches;
    r.chosen = arm;
    r.launch_projected = const_cast<void*>(projection);
    r.launch_rotary = const_cast<void*>(rotary);
    r.launch_stream = owner;
    return r.next_rc;
}
extern "C" int glm_r474_fresh_kv_pack_t64(void* k, void* v, const void* p, const void* r,
                                         size_t t, size_t h, void* s) { return launch_fake(64,k,v,p,r,t,h,s); }
extern "C" int glm_r474_fresh_kv_pack_t128(void* k, void* v, const void* p, const void* r,
                                          size_t t, size_t h, void* s) { return launch_fake(128,k,v,p,r,t,h,s); }

#include "actual_prepare.inc"

struct Request { size_t tokens, past; };
struct Prepared { rt::Tensor kv_latent, k_rot; };
std::pair<rt::Tensor, rt::Tensor> fresh(rt::Runtime& r, AttentionPrepare* prepare_,
                                        Prepared prepared, std::vector<Request> requests, size_t tokens) {
#include "actual_fresh_call.inc"
    return fresh_all;
}

Prepared tensors(size_t tokens = 1) {
    return {std::make_shared<rt::TensorImpl>(rt::Shape{tokens,512}),
            std::make_shared<rt::TensorImpl>(rt::Shape{tokens,1,64})};
}

void original(const rt::Runtime& r) {
    assert(r.project == 1 && !r.launches && r.original_copies == 5 && r.contiguous_copies == 1
           && r.syncs == 1 && r.marker_count == 2 && !r.drains);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const unsigned configured = std::stoul(argv[1]);
    AttentionPrepare prepare;
    namespace route = glm_r474_fresh_kv_route;
    unsigned fallback_cases = 0, candidate_cases = 0, thrown_cases = 0;
    {
        rt::Runtime r;
        const auto input = tensors();
        const auto output = fresh(r, &prepare, input, {{1,8496}}, 1);
        assert(output.first && output.second && route::active_mode(&r) == route::Mode::off);
        if (configured) {
            assert(r.chosen == configured && r.launches == 1 && r.project == 1 && r.syncs == 1
                   && r.launch_rotary == input.k_rot->data() && r.launch_stream == &r
                   && !r.original_copies && !r.contiguous_copies && r.marker_count == 2);
            ++candidate_cases;
        } else {
            original(r);
            ++fallback_cases;
        }
    }
    for (unsigned mode = 0; mode < 11; ++mode) {
        rt::Runtime r, other;
        auto input = tensors(mode == 2 ? 2 : 1);
        std::vector<Request> requests{{1,mode == 1 ? size_t{0} : size_t{8496}}};
        if (mode == 2) requests.push_back({1,8496});
        if (mode >= 3 && mode <= 6) r.project_fault = int(mode - 2);
        if (mode == 7) input.k_rot->strides[0] = 12;
        if (mode == 8) r.has_stream = false;
        if (mode == 9) input.k_rot->dense = false;
        if (mode == 10) input.k_rot->pointer = false;
        if (mode == 0) {
            const route::ScopedMode foreign(&other, route::Mode::tile64);
            prepare.expand_cached(r,input.kv_latent,input.k_rot);
            assert(route::active_mode(&other) == route::Mode::tile64);
        } else {
            fresh(r, &prepare, input, requests, mode == 2 ? 2 : 1);
        }
        original(r);
        ++fallback_cases;
    }
    for (unsigned fault : {0u, 1u, 2u}) {
        if (fault != 0 && !configured) continue;
        rt::Runtime r;
        r.recording = fault == 0;
        r.next_rc = fault == 1 ? 73 : 0;
        r.bad_output = fault == 2;
        try {
            fresh(r, &prepare, tensors(), {{1,8496}}, 1);
        } catch (const std::exception&) {
            assert(route::active_mode(&r) == route::Mode::off);
            assert(!r.syncs && !r.original_copies && r.project == (fault ? 1u : 0u)
                   && r.drains == (fault ? 1u : 0u) && r.launches == (fault == 1 ? 1u : 0u));
            ++thrown_cases;
            continue;
        }
        assert(false);
    }
    std::cout << "{\"fallback\":" << fallback_cases << ",\"candidate\":" << candidate_cases
              << ",\"errors\":" << thrown_cases << "}\n";
}
