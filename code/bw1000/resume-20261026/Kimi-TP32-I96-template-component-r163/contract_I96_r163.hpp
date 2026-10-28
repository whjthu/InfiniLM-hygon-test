#pragma once
#include <cstddef>
#include <cstdint>
namespace r163 {
constexpr size_t MaxT=64,E=896,TopK=16,I=96,H=3584;
constexpr size_t align256(size_t v){return (v+255)/256*256;}
constexpr size_t activation_bytes=MaxT*TopK*I*2;
constexpr size_t counts_offset=align256(activation_bytes);
constexpr size_t routes_offset=align256(counts_offset+E*4);
constexpr size_t expert_offset=align256(routes_offset+E*MaxT*TopK*4);
constexpr size_t start_offset=align256(expert_offset+MaxT*TopK*4);
constexpr size_t jobs_offset=align256(start_offset+MaxT*TopK*4);
constexpr size_t bytes=align256(jobs_offset+4);
static_assert(I%32==0&&I*32==3072&&bytes<4*1024*1024);
}
