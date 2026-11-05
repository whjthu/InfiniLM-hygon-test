#pragma once
#include <cstddef>
#include <stdexcept>
#include <algorithm>
namespace kimi_split_v128 {
constexpr std::size_t pages=132,page=256,heads=3,kdim=192,vdim=128,stride=320,context=32768,qtile=8;
struct Plan {std::size_t queries,past,total;};
constexpr std::size_t fa_query=32,fa_keys=3072;
struct Tile {std::size_t first,rows,end;};
inline Tile tile(Plan p,std::size_t first){
 if(first>=p.queries)throw std::invalid_argument("splitV querytile index");
 auto rows=std::min({fa_query,p.queries-first,fa_keys-(p.past+first)%fa_keys});return {first,rows,p.past+first+rows};
}
inline void authorize(bool selected,bool authorized,bool ar,bool vocab,int nodes,int pp,std::size_t rows){
 if(selected!=authorized||(selected&&(!ar||!vocab||nodes!=4||pp!=1||rows!=1)))throw std::invalid_argument("splitV new typed factory scope");
}
inline Plan plan(std::size_t queries,std::size_t total){
 if(!queries||queries>512||total<queries||total>context)throw std::invalid_argument("splitV C1 complete1..512/full32768");
 return {queries,total-queries,total};
}
constexpr std::size_t old_bytes=2*pages*page*heads*kdim*2;
constexpr std::size_t new_bytes=pages*page*heads*stride*2;
constexpr std::size_t per_layer_saving=old_bytes-new_bytes;
}
