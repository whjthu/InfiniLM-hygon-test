#pragma once
#include <cstddef>
#include <cstdint>
namespace glm_decode_r169 {
// Caller invokes only when observer policy is enabled. Prefix and B>1 never count.
inline bool capture(std::size_t tokens,std::size_t count,const std::size_t*rows,std::uint64_t&seen){
 if(tokens!=1||count!=1||!rows||rows[1]==0||rows[2]!=1)return false;
 ++seen;return seen>=17&&seen<=19;
}
}
