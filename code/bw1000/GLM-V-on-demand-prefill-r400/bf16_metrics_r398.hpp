#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
namespace r398_metrics {
inline double bf16(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return double(f);}
struct Accumulator {
 size_t elements=0,different=0;double err2=0,reference2=0,maxabs=0;bool finite=true;
 void add(const uint16_t*reference,const uint16_t*actual,size_t n){
  for(size_t i=0;i<n;++i){double a=bf16(reference[i]),b=bf16(actual[i]);bool ok=std::isfinite(a)&&std::isfinite(b);finite=finite&&ok;++elements;different+=reference[i]!=actual[i];if(ok){double d=b-a;err2+=d*d;reference2+=a*a;maxabs=std::max(maxabs,std::abs(d));}}
 }
 double NRMSE()const{return std::sqrt(err2/std::max(reference2,1e-300));}
 double reference_RMS()const{return elements?std::sqrt(reference2/double(elements)):0.;}
};
}
