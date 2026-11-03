#pragma once
#include <cstddef>
#include <stdexcept>
namespace kimi_r440 {
struct Geometry {std::size_t n,k,local_n,row_begin;};
inline Geometry geometry(int which,std::size_t rank,std::size_t tp) {
 if(which<0||which>1||tp!=16||rank>=tp)throw std::invalid_argument("440 exact two-role TP16 rank required");
 const std::size_t n=which?7168:3584,k=which?3584:7168;
 return {n,k,n/tp,rank*(n/tp)};
}
// Gathered rank-major [TP,1,N/TP] is exactly row-major [1,N] only for C1.
// M>1 requires a different reorder; it deliberately remains the old path.
inline bool eligible(std::size_t m,std::size_t k,std::size_t tp,int which) {
 return tp==16&&m==1&&which>=0&&which<=1&&k==(which?3584u:7168u);
}
}
