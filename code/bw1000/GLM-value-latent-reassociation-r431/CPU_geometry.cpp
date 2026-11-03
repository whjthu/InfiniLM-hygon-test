#include "geometry_r431.hpp"
#include <cassert>
#include <set>
int main(){
 size_t checked=0;for(size_t rank=0;rank<8;++rank){std::set<size_t>rows;
  for(size_t head=0;head<8;++head)for(size_t channel=0;channel<256;++channel){
   auto row=glm_value_r431::value_row(rank,head,channel);assert(row>=rank*3584&&row<(rank+1)*3584);rows.insert(row);
   for(size_t b=0;b<4;++b){assert(glm_value_r431::scale_index(rank,head,channel,b)==(row/128)*4+b);assert(glm_value_r431::scale_index(rank,head,channel,b)<896);++checked;}
   assert(glm_value_r431::diagonal_value_index(head,head,channel)==(head*8+head)*256+channel);
  }assert(rows.size()==2048);
 }assert(checked==65536);
 for(auto bad:{size_t{8},size_t{9}}){bool refused=false;try{glm_value_r431::value_row(bad,0,0);}catch(...){refused=true;}assert(refused);}
}
