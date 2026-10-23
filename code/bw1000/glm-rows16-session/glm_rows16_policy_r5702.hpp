#pragma once
#include "glm_model_resident_policy_r5640.hpp"
// Private CPU policy extension. Does not alter any Model class layout.
namespace glm_rows16_r5702 {
inline bool rows_supported(size_t rows){return rows==16||rows==64;}
inline bool resident_geometry(size_t rows,size_t capacity,size_t heads,bool host){return rows_supported(rows)&&capacity==640&&heads==8&&host;}
inline bool transient_geometry(size_t rows,int enabled){return rows_supported(rows)&&(enabled==0||(enabled==1&&rows==64));}
inline glm_model_resident_r5640::Mask mask(const uint8_t*p,size_t count,size_t begin,size_t end,size_t rows){
 if(rows==64)return glm_model_resident_r5640::mask(p,count,begin,end);
 if(rows!=16||!p||count!=78||!((begin==0&&end==38)||(begin==38&&end==78)))throw std::invalid_argument("5701 rows16 TP8PP2 mask");
 glm_model_resident_r5640::Mask result{};
 for(size_t i=0;i<78;++i){if(p[i]>1||(p[i]&&(i<begin||i>=end)))throw std::invalid_argument("5701 globalmask foreignstage/value");result[i]=p[i]!=0;}
 // Full localstage is representable, NOT automatically memory-admitted.
 return result;
}
inline uint64_t admission_bytes(size_t rows,size_t selected){
 if(!rows_supported(rows)||selected>40)throw std::invalid_argument("5702 rows/layer count");
 // Pooled40MiB K and V each physically rounded64MiB in actual460.
 // No exactparameter-bank memory saving is assumed by this candidate.
 return selected*(rows==16?128ull*1024*1024:glm_model_resident_r5640::bank_bytes);
}
} // namespace glm_rows16_r5702
