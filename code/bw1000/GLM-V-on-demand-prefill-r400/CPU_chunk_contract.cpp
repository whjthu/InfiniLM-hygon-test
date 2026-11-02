#include "chunk_geometry_r400.hpp"
#include <vector>
#include <cassert>
#include <iostream>
int main(){
 using namespace infinilm::models::glm_moe_dsa::v_on_demand_r400;
 for(size_t U:{size_t(1),size_t(512),size_t(2048),size_t(8192),size_t(8193),size_t(32768)}){
  std::vector<size_t>out(U,size_t(-1));size_t calls=0,total=0;
  for_blocks(U,[&](size_t begin,size_t count){assert(count&&count<=8192&&begin==total);++calls;for(size_t j=0;j<count;++j)out[begin+j]=begin+j;total+=count;});
  assert(total==U&&calls==(U+8191)/8192);for(size_t i=0;i<U;++i)assert(out[i]==i);
 }
 for(size_t U:{size_t(0),size_t(32769)}){bool bad=false;try{for_blocks(U,[](size_t,size_t){});}catch(...){bad=true;}assert(bad);}
 std::cout<<"{\"passed\":true,\"actual_shared_chunk_loop\":true,\"max_rows\":8192,\"U32768_calls\":4,\"U2048_calls\":1,\"boundaries_and_order\":true,\"GPU\":0,\"Native_math_qualified\":false}\n";
}
