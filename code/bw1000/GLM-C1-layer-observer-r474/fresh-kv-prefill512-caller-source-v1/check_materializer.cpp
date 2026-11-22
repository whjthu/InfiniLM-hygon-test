#include "fresh_kv_prefill512.hpp"
#include "fresh_kv_state.inc"

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
namespace prefill = glm_r474_fresh_kv_prefill512;
namespace tf = prefill::layout;
using Counts = std::array<std::size_t, prefill::sections>;

std::size_t pack_calls = 0;
std::size_t old_copy = 0;
std::size_t old_contiguous = 0;

namespace rt {
enum class DataType { kBFloat16, wrong };

struct Runtime {
    void* stream_value{this};
    void* stream() const { return stream_value; }
};

struct TensorFake;
using Tensor = std::shared_ptr<TensorFake>;
struct TensorFake : std::enable_shared_from_this<TensorFake> {
    tf::TensorLayout layout;
    std::size_t physical_bytes;
    std::size_t element_bytes{2};
    std::uintptr_t pointer;
    DataType dtype_value{DataType::kBFloat16};
    int device_value{0};
    bool contiguous_value{true};

    TensorFake(tf::TensorLayout spec, std::uintptr_t address)
        : layout(spec), physical_bytes(tf::bytes(spec)), pointer(address) {}
    int device() const { return device_value; }
    DataType dtype() const { return dtype_value; }
    std::size_t element_size() const { return element_bytes; }
    std::size_t nbytes() const { return physical_bytes; }
    std::size_t ndim() const { return layout.shape.size(); }
    bool is_contiguous() const { return contiguous_value; }
    std::size_t size(std::size_t i) const { return layout.shape.at(i); }
    std::size_t stride(std::size_t i) const { return layout.stride.at(i); }
    std::byte* data() const { return reinterpret_cast<std::byte*>(pointer); }
    Tensor narrow(std::initializer_list<std::initializer_list<std::size_t>>) { return shared_from_this(); }
    void copy_from(Runtime&, const Tensor&) { ++old_copy; }
    Tensor contiguous(Runtime&) { ++old_contiguous; return shared_from_this(); }
};
}

struct Fixture {
    rt::Runtime runtime;
    rt::Tensor projected{std::make_shared<rt::TensorFake>(tf::projected, 0x10000000)};
    rt::Tensor rotary{std::make_shared<rt::TensorFake>(tf::rotary, 0x11000000)};
    rt::Tensor key{std::make_shared<rt::TensorFake>(tf::output, 0x12000000)};
    rt::Tensor output{std::make_shared<rt::TensorFake>(tf::output, 0x13000000)};
    bool fail_enqueue{false};
};

Fixture* executing = nullptr;

extern "C" int glm_r474_fresh_kv_pack_t64(void*, void*, const void*, const void*, size_t, size_t, void*) {
    assert(false);
    return 1;
}

extern "C" int glm_r474_fresh_kv_pack_t128(void* key, void* value, const void* expanded,
                                         const void* rotary, size_t tokens, size_t heads, void* stream) {
    assert(executing && key == executing->key->data() && value == executing->output->data());
    assert(expanded == executing->projected->data() && rotary == executing->rotary->data());
    assert(tokens == 512 && heads == 4 && stream == executing->runtime.stream());
    ++pack_calls;
    return executing->fail_enqueue;
}

rt::Tensor materialize(Fixture& fixture) {
    auto& r = fixture.runtime;
    auto& expanded = fixture.projected;
    auto& rotary = fixture.rotary;
    auto& key = fixture.key;
    const size_t tokens = 512;
    const size_t heads_ = 4;
    const int device_ = 0;
    struct Work {
        Fixture& fixture;
        std::vector<rt::Tensor> held;
        rt::Tensor alloc(std::initializer_list<std::size_t> shape, rt::DataType dtype) {
            assert(shape.size() == 3 && dtype == rt::DataType::kBFloat16);
            held.push_back(fixture.output);
            return fixture.output;
        }
    } work{fixture, {}};
    #include "fresh_kv_materialize.inc"
    return value;
}

struct Context {
    struct Inner {
        bool ready{true};
        std::shared_ptr<const rt::Runtime> owner;
    };
    std::shared_ptr<Inner> inner;
    Fixture fixture;
};

Context& global470(void* pointer) { return *static_cast<Context*>(pointer); }
std::size_t bridged = 0;
std::size_t errors = 0;
void report470(const std::exception&) { ++errors; }
int modern_glm_global_forward_r470(void* handle, size_t, size_t, const size_t*,
                                  const void*, const void*, void*, int, void*) {
    auto& ctx = global470(handle);
    ++bridged;
    executing = &ctx.fixture;
    try {
        prefill::with_fresh(&ctx.fixture.runtime, true, [&]{ return materialize(ctx.fixture); });
        assert(!prefill::active_for(&ctx.fixture.runtime));
        return 0;
    } catch (const std::runtime_error&) {
        return 1;
    }
}
#include "forward_explicit_prefill512.inc"

std::array<size_t,3> request{0,0,512};

int launch(Context& ctx, Counts& counts, std::size_t tokens=512, std::size_t count=1,
           const std::size_t* packet=request.data(), std::size_t words=prefill::sections, bool stream=true) {
    return modern_glm_global_fresh_kv_prefill512_r474(&ctx, tokens, count, packet, nullptr, nullptr,
        nullptr, 0, stream ? ctx.fixture.runtime.stream() : nullptr, counts.data(), words);
}

Context make_with_owner() {
    return Context{};
}

void own(Context& ctx) {
    ctx.inner = std::make_shared<Context::Inner>();
    ctx.inner->owner = std::shared_ptr<const rt::Runtime>(&ctx.fixture.runtime, [](const rt::Runtime*){});
}

template<class Change>
void rejection(Change change, prefill::Counter counter) {
    auto ctx = make_with_owner();
    own(ctx);
    change(ctx.fixture);
    Counts observed{};
    Counts expected{};
    expected[prefill::attempted] = 1;
    expected[counter] = 1;
    const auto before = pack_calls;
    const auto baseline = old_copy;
    const auto contiguous_copy = old_contiguous;
    assert(launch(ctx, observed) == 0 && observed == expected);
    assert(pack_calls == before && old_copy == baseline + 5 && old_contiguous == contiguous_copy + 1);
}

int main() {
    static_assert(tf::bytes(tf::projected) == 1835008);
    static_assert(tf::bytes(tf::rotary) == 65536);
    static_assert(tf::bytes(tf::output) == 1048576);
    auto ctx = make_with_owner();
    own(ctx);
    assert(materialize(ctx.fixture) == ctx.fixture.projected && pack_calls == 0 && old_copy == 5);
    Counts counts{};
    assert(launch(ctx, counts) == 0 && (counts == Counts{1,1,0,0,0}));
    assert(pack_calls == 1 && old_copy == 5 && !prefill::active_for(&ctx.fixture.runtime));

    rejection([](Fixture& f){ f.rotary->layout.stride[0] += 64; }, prefill::rotary_layout);
    rejection([](Fixture& f){ f.rotary->layout.shape[1] = 4; }, prefill::rotary_layout);
    rejection([](Fixture& f){ f.rotary->dtype_value = rt::DataType::wrong; }, prefill::rotary_layout);
    rejection([](Fixture& f){ f.rotary->physical_bytes -= 2; }, prefill::rotary_layout);
    rejection([](Fixture& f){ f.projected->layout.stride[1] = 512; }, prefill::tensor_layout);
    rejection([](Fixture& f){ f.projected->device_value = 1; }, prefill::tensor_layout);
    rejection([](Fixture& f){ f.key->element_bytes = 4; }, prefill::tensor_layout);
    rejection([](Fixture& f){ f.output->contiguous_value = false; }, prefill::tensor_layout);
    rejection([](Fixture& f){ f.output->pointer = f.key->pointer + 16; }, prefill::address_span);
    rejection([](Fixture& f){ f.rotary->pointer += 1; }, prefill::address_span);
    rejection([](Fixture& f){ f.projected->pointer = std::numeric_limits<std::uintptr_t>::max()-1; }, prefill::address_span);

    const std::size_t original_calls = bridged;
    for (auto packet : {std::array<size_t,3>{1,0,512}, {0,1,512}, {0,0,511}}) {
        assert(launch(ctx, counts, 512, 1, packet.data()) == 1 && counts == Counts{});
    }
    assert(launch(ctx, counts, 1) == 1 && counts == Counts{});
    assert(launch(ctx, counts, 512, 2) == 1 && counts == Counts{});
    assert(launch(ctx, counts, 512, 1, nullptr) == 1 && counts == Counts{});
    assert(launch(ctx, counts, 512, 1, request.data(), 4) == 1 && counts == Counts{});
    assert(launch(ctx, counts, 512, 1, request.data(), 5, false) == 1 && counts == Counts{});
    assert(bridged == original_calls && errors == 8);

    ctx.fixture.fail_enqueue = true;
    assert(launch(ctx, counts) == 1 && (counts == Counts{1,0,0,0,0}));
    assert(!prefill::active_for(&ctx.fixture.runtime));
    std::cout << "initial trial, ordinary fallback, 11 tensor/address rejections, 8 bridge rejections, and one enqueue-fail count\n";
}
