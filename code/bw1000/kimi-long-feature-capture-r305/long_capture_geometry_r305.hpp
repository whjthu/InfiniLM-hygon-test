#pragma once
#include <cstddef>
#include <cstdint>
#include <climits>
#include <set>
#include <stdexcept>
namespace kimi_long_features_r305 {
struct ContextGeometry {
 bool long_agent,geometry_long,ready;
 size_t logical,physical,pool,context,pages_per_row,blocks,block_size,cache_rows;
 int tp_nodes,pp_size,stage;
};
struct Scope {
 size_t context=1024,pages=4,logical_cap=32,max_width=1024,max_total=8192;
 bool scoped_long=false;
};
inline Scope long_scope(){return {32768,128,1,512,512,true};}
inline Scope legacy_scope(){return {};}
inline void validate_scope(const Scope&s){
 if(s.scoped_long){if(s.context!=32768||s.pages!=128||s.logical_cap!=1||s.max_width!=512||s.max_total!=512)throw std::invalid_argument("r305 long capture scope");}
 else if(s.context!=1024||s.pages!=4||s.logical_cap!=32||s.max_width!=1024||s.max_total!=8192)throw std::invalid_argument("r305 legacy capture scope");
}
inline Scope validate_context(const ContextGeometry&g){
 if(!g.long_agent||!g.geometry_long||!g.ready||g.logical!=1||g.physical!=2||g.pool!=3||g.context!=32768||g.pages_per_row!=128||g.blocks!=260||g.block_size!=256||g.cache_rows!=2||g.tp_nodes!=2||g.pp_size!=2||g.stage<0||g.stage>=2)throw std::invalid_argument("r305 requires Ready actual TP16PP2 long Context");
 return long_scope();
}
inline size_t validate_metadata(const Scope&s,uint64_t epoch,size_t count,const int32_t*logical,const int32_t*past,const int32_t*width){
 validate_scope(s);if(!epoch||epoch>uint64_t(INT64_MAX)||!count||count>s.logical_cap||!logical||!past||!width)throw std::invalid_argument("r305 capture metadata header");
 size_t total=0;std::set<int32_t>seen;
 for(size_t i=0;i<count;++i){
  if(logical[i]<1||size_t(logical[i])>s.logical_cap||!seen.insert(logical[i]).second||past[i]<0||width[i]<1||size_t(width[i])>s.max_width||int64_t(past[i])+int64_t(width[i])>int64_t(s.context))throw std::invalid_argument("r305 logical/past/width/context");
  total+=size_t(width[i]);
 }
 if(total>s.max_total)throw std::invalid_argument("r305 total capture tokens");return total;
}
inline void validate_observed(const Scope&s,uint64_t epoch,size_t count,const int32_t*logical,const int32_t*past,const int32_t*width,const int32_t*actual_past,const int32_t*actual_total,const int32_t*offset,const int32_t*table){
 const auto total=validate_metadata(s,epoch,count,logical,past,width);if(!actual_past||!actual_total||!offset||!table)throw std::invalid_argument("r305 observed metadata pointers");size_t cursor=0;
 for(size_t i=0;i<count;++i){
  if(actual_past[i]!=past[i]||int64_t(actual_total[i])-actual_past[i]!=width[i]||offset[i]!=int32_t(cursor))throw std::invalid_argument("r305 observed past/width/offset");
  cursor+=size_t(width[i]);
  for(size_t j=0;j<s.pages;++j)if(table[i*s.pages+j]!=1+(logical[i]-1)*int32_t(s.pages)+int32_t(j))throw std::invalid_argument("r305 logical MLA KV pages; never KDA state rows");
 }
 if(offset[count]!=int32_t(total))throw std::invalid_argument("r305 observed endpoint");
}
}
