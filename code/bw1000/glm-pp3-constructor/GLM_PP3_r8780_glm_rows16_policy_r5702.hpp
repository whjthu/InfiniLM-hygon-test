#pragma once
#include "glm_model_resident_policy_r5640.hpp"
namespace glm_rows16_r5702 {
inline bool rows_supported(size_t rows){return rows==64;}
inline bool resident_geometry(size_t rows,size_t capacity,size_t heads,bool host){return rows==64&&capacity==640&&heads==8&&host;}
inline bool transient_geometry(size_t rows,int enabled){return rows==64&&enabled==0;}
inline glm_model_resident_r5640::Mask mask(const uint8_t*p,size_t count,size_t begin,size_t end,size_t rows){if(rows!=64)throw std::invalid_argument("8780 rows64 only");return glm_model_resident_r5640::mask(p,count,begin,end);}
inline uint64_t admission_bytes(size_t rows,size_t selected){if(rows!=64||selected>28)throw std::invalid_argument("8780 PP3 rows/layers");return selected*512ull*1024*1024;}
}
