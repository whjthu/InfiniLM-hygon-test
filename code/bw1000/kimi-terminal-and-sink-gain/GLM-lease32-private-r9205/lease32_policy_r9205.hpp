#pragma once
#include <cstddef>
#include <array>
#include <algorithm>
#include <cstdint>
namespace glm_lease32_r9205 {inline bool use(bool base,bool optin,size_t tokens,size_t count,bool all_positions){return base&&tokens==count&&(count<=16||(optin&&count<=32))&&!all_positions;}}
