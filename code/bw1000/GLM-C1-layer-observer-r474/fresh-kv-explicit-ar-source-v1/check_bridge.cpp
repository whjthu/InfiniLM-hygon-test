#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

#include "fresh_kv_explicit_ar.hpp"

bool seen_from_attention(const void* owner);

struct Runtime {};
struct Inner {
    std::shared_ptr<Runtime> owner = std::make_shared<Runtime>();
    bool ready = true;
};
struct Context {
    std::unique_ptr<Inner> inner = std::make_unique<Inner>();
};

Context& global470(void* handle) {
    if (!handle) throw std::invalid_argument("no context");
    return *static_cast<Context*>(handle);
}

thread_local const char* error_kind = nullptr;

void report470(const std::exception& error) {
    error_kind = std::strcmp(error.what(), "forward failed") == 0 ? "forward" : "boundary";
}

thread_local bool expect_decode = false;
int calls = 0;
int forward_status = 0;
bool throw_in_forward = false;
Runtime other;

extern "C" int modern_glm_global_forward_r470(
    void* handle, size_t tokens, size_t count, const size_t* requests,
    const void* input, const void* positions, void* logits, int all_positions, void* stream);

#include "forward_explicit_ar.inc"

extern "C" int modern_glm_global_forward_r470(
    void* handle, size_t tokens, size_t count, const size_t* requests,
    const void* input, const void* positions, void* logits, int all_positions, void* stream) {
    ++calls;
    auto* owner = global470(handle).inner->owner.get();
    assert(seen_from_attention(owner) == expect_decode);
    assert(!seen_from_attention(&other));
    assert(tokens == 1 && count == 1 && requests[1] == 512 && all_positions == 0);
    assert(input == &other && positions == &other && logits == &other && stream == &other);
    if (expect_decode) {
        auto rc = modern_glm_global_fresh_kv_decode_r474(handle, tokens, count, requests, input,
                                                        positions, logits, all_positions, stream);
        assert(rc == 1 && seen_from_attention(owner));
        std::thread second_thread([owner] { assert(!seen_from_attention(owner)); });
        second_thread.join();
    }
    if (throw_in_forward) throw std::runtime_error("forward failed");
    return forward_status;
}

int main() {
    Context context;
    auto* owner = context.inner->owner.get();
    const size_t valid[]{0, 512, 1};
    const size_t first[]{0, 0, 1};
    const size_t foreign_row[]{1, 512, 1};
    const size_t bad_length[]{0, 512, 2};
    const auto decode = [](void* handle, size_t tokens, size_t count, const size_t* packet, int all) {
        return modern_glm_global_fresh_kv_decode_r474(handle, tokens, count, packet, &other, &other,
                                                     &other, all, &other);
    };

    assert(modern_glm_global_forward_r470(&context, 1, 1, valid, &other, &other, &other, 0, &other) == 0);
    assert(calls == 1 && !seen_from_attention(owner));

    expect_decode = true;
    assert(decode(&context, 1, 1, valid, 0) == 0);
    assert(calls == 2 && !seen_from_attention(owner));
    forward_status = 7;
    assert(decode(&context, 1, 1, valid, 0) == 7);
    assert(calls == 3 && !seen_from_attention(owner));
    throw_in_forward = true;
    assert(decode(&context, 1, 1, valid, 0) == 1);
    assert(calls == 4 && !seen_from_attention(owner) && std::strcmp(error_kind, "forward") == 0);
    assert(decode(nullptr, 1, 1, valid, 0) == 1);
    assert(decode(&context, 1, 1, nullptr, 0) == 1);
    assert(decode(&context, 1, 1, valid, 1) == 1);
    assert(decode(&context, 2, 1, valid, 0) == 1);
    assert(decode(&context, 1, 2, valid, 0) == 1);
    assert(decode(&context, 1, 1, first, 0) == 1);
    assert(decode(&context, 1, 1, foreign_row, 0) == 1);
    assert(decode(&context, 1, 1, bad_length, 0) == 1);
    context.inner->ready = false;
    assert(decode(&context, 1, 1, valid, 0) == 1);
    assert(calls == 4 && !seen_from_attention(owner));
}
