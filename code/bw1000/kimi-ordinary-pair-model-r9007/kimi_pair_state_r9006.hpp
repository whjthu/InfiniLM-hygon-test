#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace kimi_pair_r9006 {
struct Geometry {
 size_t logical,physical,half;
 explicit Geometry(size_t b):logical(b),physical(std::max(size_t(16),2*b)),half(physical/2){
  if(!b||b>64)throw std::invalid_argument("9006 logical requests1..64");
 }
 size_t state_rows()const{return physical+1;}
 size_t pages()const{return 4*(physical+1);}
 // H6,D128,BF16 KDA matrix and packed K4 conv history, per local KDA layer.
 size_t recurrent_bytes_per_layer()const{return state_rows()*(6*128*128+3*6*128*3)*2;}
 size_t extra_vs_ordinary16_per_layer()const{return (physical-16)*(6*128*128+3*6*128*3)*2;}
 size_t kv_bytes_per_MLA_layer()const{return pages()*256*(512+64)*2;}
};
struct Row {int active,shadow,past=0,phase=0;size_t width=0,retained=0;std::vector<int64_t> ids;};
// phase:0 unprefilled,1 ready,2 verified,3 replay required,4 terminal.
class State {
 const void*owner_;bool poisoned_=false,ordinary_=false;uint64_t generation_=0;std::vector<Row> rows_;
 public: Geometry geometry;
 State(const void*owner,size_t b):owner_(owner),geometry(b){if(!owner)throw std::invalid_argument("9006 owner");rows_.reserve(b);for(size_t k=1;k<=b;++k){rows_.push_back({int(k),int(k+geometry.half)});rows_.back().ids.reserve(8);}}
 void owner(const void*p)const{if(p!=owner_||poisoned_)throw std::invalid_argument("9006 wrong/poisoned owner");}
 Row&row(const void*p,int k){owner(p);if(ordinary_)throw std::invalid_argument("9007 ordinary mode excludes pair API");if(k<1||size_t(k)>rows_.size())throw std::invalid_argument("9006 logical row");return rows_[size_t(k-1)];}
 void preflight_prefill(const void*p,int k,size_t n){auto&r=row(p,k);if(r.phase||!n||n>1024)throw std::invalid_argument("9006 prefill once, width1..1024");}
 void prefilled(const void*p,int k,size_t n){preflight_prefill(p,k,n);auto&r=row(p,k);r.past=int(n);r.phase=1;}
 void check(const void*p,int k,int past,int read,int write,const std::vector<int64_t>&ids) {
  auto&r=row(p,k);if((r.phase!=1&&r.phase!=3)||past!=r.past||read!=r.active||write!=r.shadow||read==write||ids.empty()||ids.size()>8||size_t(past)+ids.size()>1024)throw std::invalid_argument("9006 phase/length/pair/width");
  if(r.phase==3&&(ids.size()!=r.retained||!std::equal(ids.begin(),ids.end(),r.ids.begin())))throw std::invalid_argument("9006 replay must equal retained verify prefix");
 }
 void completed(const void*p,int k,const std::vector<int64_t>&ids){auto&r=row(p,k);check(p,k,r.past,r.active,r.shadow,ids);if(r.phase==3){r.past+=int(ids.size());std::swap(r.active,r.shadow);r.phase=1;r.ids.clear();}else{r.ids=ids;r.width=ids.size();r.phase=2;}}
 void commit(const void*p,int k,size_t accepted,bool continuation){auto&r=row(p,k);if(r.phase!=2||accepted>=r.width)throw std::invalid_argument("9006 accepted must0..width-1");r.retained=accepted+1;
  if(!continuation){r.phase=4;return;}if(r.retained<r.width){r.phase=3;return;}r.past+=int(r.retained);std::swap(r.active,r.shadow);r.phase=1;r.ids.clear();
 }
 uint64_t generation()const{return generation_;}
 void ordinary(const void*p,uint64_t gen)const{owner(p);if(!ordinary_||gen!=generation_)throw std::invalid_argument("9007 ordinary mode/generation");}
 void switch_mode(const void*p,bool ordinary){owner(p);for(size_t j=0;j<rows_.size();++j){auto&r=rows_[j];r.active=int(j+1);r.shadow=int(j+1+geometry.half);r.past=r.phase=0;r.width=r.retained=0;r.ids.clear();}ordinary_=ordinary;++generation_;}
 void poison(){poisoned_=true;}
};
}
