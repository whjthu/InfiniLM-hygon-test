#pragma once
#include <cmath>
#ifdef __CUDACC__
#define KIMI_SPLIT_HD __host__ __device__
#else
#define KIMI_SPLIT_HD
#endif
namespace kimi_split_v128 {
struct Factors{float a,b,maximum,denominator;};
KIMI_SPLIT_HD inline Factors factors(float oldmax,float oldden,float lse,bool first){
 float maximum=first?lse:fmaxf(oldmax,lse);float a=first?0.f:expf(oldmax-maximum),b=expf(lse-maximum);
 return {a,b,maximum,(first?0.f:oldden*a)+b};
}
KIMI_SPLIT_HD inline float merge_value(float oldsum,float partial,float a,float b,bool first){return(first?0.f:oldsum*a)+partial*b;}
}
#undef KIMI_SPLIT_HD
