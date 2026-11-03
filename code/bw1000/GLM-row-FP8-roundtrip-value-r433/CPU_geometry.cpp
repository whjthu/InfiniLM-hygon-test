#include "geometry_r433.hpp"
#include <cassert>
#include <iostream>
#include <set>
int main(){
 using namespace glm_value_r433;size_t diagonal=0;
 for(size_t i=0;i<512;++i)for(size_t j=0;j<512;++j){auto v=identity_code(i,j);assert(v==(i==j?0x38:0));diagonal+=v==0x38;}
 assert(diagonal==512&&value_weight_bytes==4194304&&query_identity_FLOPs==1073741824&&setup_identity_FLOPs==1879048192);
 std::set<size_t>global_rows;size_t addresses=0;
 for(size_t rank=0;rank<8;++rank)for(size_t h=0;h<8;++h)for(size_t v=0;v<256;++v){
  auto row=row_offset(rank)+value_column(h)+v;assert(row==value_row(rank,h,v)&&row<28672);assert(global_rows.insert(row).second);
  for(size_t k=0;k<4;++k){assert(scale_index(rank,h,v,k)==(row/128)*4+k&&scale_index(rank,h,v,k)<896);++addresses;}
 }
 assert(addresses==65536&&global_rows.size()==16384);
 auto rejects=[](auto f){try{f();}catch(const std::invalid_argument&){return true;}return false;};
 assert(rejects([]{identity_code(512,0);})&&rejects([]{identity_code(0,512);})&&rejects([]{row_offset(8);})&&rejects([]{value_column(8);})&&rejects([]{scale_index(0,0,0,4);}));
 std::cout<<"r433 actual identity/head/scale helpers PASS; no Runtime/device/numeric oracle\n";
}
