#pragma once
#include <cstddef>
namespace glm_selected_plan_r426 {
inline bool fits(std::size_t dtype,std::size_t m,std::size_t n,std::size_t k,
                 std::size_t sr,std::size_t sc,std::size_t ro,std::size_t co){
 return dtype==1&&m==2048&&n==3584&&k==512&&sr==224&&sc==4&&co==0&&
        ro<28672&&ro%3584==0;
}
}
