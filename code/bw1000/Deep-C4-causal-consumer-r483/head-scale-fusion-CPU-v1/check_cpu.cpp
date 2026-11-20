#include "cpu_fixture_runtime.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

using Leaf = int(void*, const void*, const void*, const void*, size_t, size_t, void*);
extern "C" Leaf deep_c4_head_scale_i1_v1;
extern "C" Leaf deep_c4_head_scale_i4_v1;

static uint32_t bits(float value) {
    uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

// Model uses the stored float32 of 1/sqrt(8192). With the same FP32 operations
// written as separate passes, compare an independently decomposed SDK oracle.
static void fill_oracle(std::vector<float>& expected, const std::vector<uint16_t>& head,
                        const std::vector<float>& query_scale, float scalar, size_t count) {
    std::vector<float> cast(count), scaled(count);
    for (size_t i=0; i<count; ++i) {
        uint32_t raw = uint32_t(head[i+1]) * 65536u;
        std::memcpy(&cast[i], &raw, sizeof(raw));
    }
    for (size_t i=0; i<count; ++i) {
        volatile float stored = cast[i] * scalar;
        scaled[i] = stored;
    }
    for (size_t i=0; i<count; ++i) {
        volatile float stored = scaled[i] * query_scale[i+1];
        expected[i] = stored;
    }
}

int main() {
    unsigned long long values=0, order_sensitive=0;
    unsigned cases=0, rejects=0;
    int stream=0;
    float scalar = float(1/std::sqrt(8192.));
    assert(bits(scalar)==0x3c3504f3u);
    constexpr std::array<uint16_t,8> edge{0x0000,0x8000,0x0001,0x8001,0x0080,0x8080,0x3f80,0xbf80};
    for (auto entry : {std::pair<Leaf*, unsigned>{deep_c4_head_scale_i1_v1,1},
                       {deep_c4_head_scale_i4_v1,4}}) {
        auto invoke=entry.first;
        for (size_t tokens : {size_t{1},size_t{15},size_t{17},size_t{511},size_t{512}}) {
            for (size_t heads : {size_t{8},size_t{64}}) {
                const size_t count=tokens*heads;
                std::vector<uint16_t> head(count+2,0x18a7);
                std::vector<float> query_scale(count+2,12345.f), result(count+2,-9876.f), expected(count);
                unsigned generator=0xb82b3307;
                for (size_t i=0; i<count; ++i) {
                    generator=1664525u*generator+1013904223u;
                    uint16_t raw=uint16_t(generator>>16);
                    if ((raw & 0x7f80u) > 0x7780u) raw ^= 0x2000u;
                    head[i+1]=i<edge.size()?edge[i]:raw;
                    query_scale[i+1]=std::ldexp(float(128+(generator&127u)), int((generator>>7)&31u) - 27);
                }
                fill_oracle(expected,head,query_scale,scalar,count);
                const auto frozen_head=head;
                const auto frozen_scale=query_scale;
                auto start=launch_count;
                last_error=73;
                assert(invoke(result.data()+1,head.data()+1,&scalar,query_scale.data()+1,
                              tokens,heads,&stream)==73);
                assert(launch_count==start+1 && last_stream==&stream && last_threads==128);
                const auto unit=128*entry.second;
                assert(last_grid==(count+unit-1)/unit);
                assert(result.front()==-9876.f && result.back()==-9876.f);
                assert(head.front()==0x18a7 && head.back()==0x18a7);
                assert(query_scale.front()==12345.f && query_scale.back()==12345.f);
                assert(head==frozen_head && query_scale==frozen_scale);
                for (size_t i=0; i<count; ++i) {
                    assert(std::isfinite(expected[i]));
                    assert(bits(result[i+1])==bits(expected[i]));
                    const float h=__uint_as_float(uint32_t(head[i+1])<<16);
                    volatile float wrong_first=scalar*query_scale[i+1];
                    volatile float wrong_order=h*wrong_first;
                    order_sensitive += bits(wrong_order) != bits(expected[i]);
                }
                values+=count;
                ++cases;
            }
        }
        float out=111,scale=0.5f;
        uint16_t head=0x3f80;
        auto start=launch_count;
        assert(invoke(nullptr,&head,&scalar,&scale,1,8,nullptr)==1);
        assert(invoke(&out,nullptr,&scalar,&scale,1,8,nullptr)==1);
        assert(invoke(&out,&head,nullptr,&scale,1,8,nullptr)==1);
        assert(invoke(&out,&head,&scalar,nullptr,1,8,nullptr)==1);
        assert(invoke(&scale,&head,&scalar,&scale,1,8,nullptr)==1);
        assert(invoke(&out,&head,&scalar,&scale,0,8,nullptr)==1);
        assert(invoke(&out,&head,&scalar,&scale,513,8,nullptr)==1);
        assert(invoke(&out,&head,&scalar,&scale,1,16,nullptr)==1);
        assert(out==111 && launch_count==start);
        rejects+=8;
    }
    assert(order_sensitive>0);
    std::cout<<"{\"cases\":"<<cases<<",\"exact_values\":"<<values<<",\"rejected_launch_inputs\":"<<rejects
             <<",\"reassociation_would_change_values\":"<<order_sensitive<<"}\n";
}
