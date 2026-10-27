#pragma once
#include <cstddef>
namespace glm_m15_oproj_r125 {
inline bool fits(std::size_t m,std::size_t n,std::size_t k,std::size_t sr,std::size_t sc,std::size_t ro,std::size_t co) noexcept {
 return m==15&&n==6144&&k==2048&&sr==48&&sc==128&&ro==0&&co<=14336&&co%2048==0;
}
constexpr std::size_t stats_count=17;
constexpr std::size_t private_geometry_workspace_bytes=31125504;
}
