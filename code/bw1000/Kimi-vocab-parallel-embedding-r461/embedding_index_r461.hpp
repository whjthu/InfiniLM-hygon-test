#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define K461_HD __host__ __device__
#else
#define K461_HD
#endif
namespace kimi_vocab_r461 {
constexpr std::size_t vocabulary=163840, hidden=7168, parts=32, local_rows=5120;
constexpr std::size_t full_bytes=vocabulary*hidden*2, local_bytes=local_rows*hidden*2;
constexpr std::size_t saved_bytes=full_bytes-local_bytes;
K461_HD inline bool valid_id(std::int64_t id) { return id>=0 && id<std::int64_t(vocabulary); }
K461_HD inline bool owns(std::int64_t id,std::size_t rank) {
    return valid_id(id) && rank<parts && std::uint64_t(id)>=rank*local_rows && std::uint64_t(id)<(rank+1)*local_rows;
}
K461_HD inline std::size_t local_index(std::int64_t id,std::size_t rank,std::size_t column) {
    return (std::size_t(id)-rank*local_rows)*hidden+column;
}
inline void validate_global_ids(const std::int64_t *ids,std::size_t tokens) {
    if(!ids||!tokens||tokens>512)throw std::invalid_argument("r461 full C1 prompt/decode interval");
    for(std::size_t i=0;i<tokens;++i)
        if(!valid_id(ids[i]))throw std::invalid_argument("r461 global token outside original vocabulary");
}
static_assert(full_bytes==2348810240ULL && local_bytes==73400320ULL && saved_bytes==2275409920ULL);
}
#undef K461_HD
