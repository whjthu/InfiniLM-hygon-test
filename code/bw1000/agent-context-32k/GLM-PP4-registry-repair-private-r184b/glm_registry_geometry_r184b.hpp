#pragma once
#include "glm_PP_layout_r3854.hpp"
#include <set>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa::registry_geometry_r184b {
struct Span{size_t begin,end;};
inline Span span(int pp,int stage){
 if((pp!=2&&pp!=4)||stage<0||stage>=pp)throw std::invalid_argument("r184b registry supports exact PP2 or PP4 topology");
 const auto b=PP_layout_r3854::boundaries(size_t(pp));return{b.at(size_t(stage)),b.at(size_t(stage+1))};
}
inline void member(int tp,int pp,int stage,size_t layer){
 const auto b=span(pp,stage);if(tp!=8||layer<b.begin||layer>=b.end)throw std::invalid_argument("r184b exact78layer TP8 PP2/PP4 identity");
}
inline void complete(int pp,int stage,const std::set<size_t>&layers){
 const auto b=span(pp,stage);if(layers.size()!=b.end-b.begin)throw std::logic_error("5775 incomplete stage identity");
 for(size_t l=b.begin;l<b.end;++l)if(!layers.count(l))throw std::logic_error("5775 incomplete stage identity");
}
inline void policy(int pp,int mode,int diagnostic,int shadow){
 if(pp!=2&&(mode!=0||diagnostic!=0||shadow!=-1))throw std::invalid_argument("r184b PP2 cross-request optimization/diagnostics remain disabled for PP4");
}
}
