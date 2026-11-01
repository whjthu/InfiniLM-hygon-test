#pragma once
#include <vector>
#include <cstddef>
namespace layout_r345 {
inline bool accept(const std::vector<size_t>&shape,const std::vector<ptrdiff_t>&stride,const std::vector<size_t>&expected){
 if(shape!=expected||stride.size()!=shape.size())return false;
 ptrdiff_t want=1;
 for(size_t j=shape.size();j-->0;){if(stride[j]<=0)return false;if(shape[j]>1&&stride[j]!=want)return false;want*=shape[j];}
 return true;
}
}
