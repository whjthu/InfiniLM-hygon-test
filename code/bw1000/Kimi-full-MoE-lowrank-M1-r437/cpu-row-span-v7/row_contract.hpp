#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace kimi_r437_row {
inline bool input_row(size_t rows,size_t columns,ptrdiff_t rowpitch,ptrdiff_t inner,size_t k){return rows==1&&columns==k&&rowpitch>0&&inner==1;}
inline bool weight(size_t rows,size_t columns,ptrdiff_t rowpitch,ptrdiff_t inner,size_t n,size_t k){return rows==n&&columns==k&&rowpitch==static_cast<ptrdiff_t>(k)&&inner==1;}
inline size_t bytes(size_t k){if(k>std::numeric_limits<size_t>::max()/2)throw std::overflow_error("437 BF16 span overflow");return k*2;}
inline bool within(uintptr_t p,uintptr_t origin,size_t available,size_t needed){return p>=origin&&p-origin<=available&&needed<=available-(p-origin);}
}
