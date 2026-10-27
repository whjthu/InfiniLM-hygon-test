#include "glm_m15_oproj_geometry_r125.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>
int main(){using glm_m15_oproj_r125::fits;int valid=0,invalid=0;
 for(size_t rank=0;rank<8;++rank){assert(fits(15,6144,2048,48,128,0,rank*2048));++valid;}
 for(size_t m=1;m<=64;++m)if(m!=15){assert(!fits(m,6144,2048,48,128,0,0));++invalid;}
 for(size_t co:{size_t(1),size_t(128),size_t(2047),size_t(14337),size_t(16384)}){assert(!fits(15,6144,2048,48,128,0,co));++invalid;}
 assert(!fits(15,6144,256,48,16,0,0));assert(!fits(15,2048,2048,128,16,14336,0));assert(!fits(15,6144,2048,48,128,1,0));assert(!fits(15,6144,2048,47,128,0,0));assert(!fits(15,6144,2048,48,127,0,0));invalid+=5;
 std::printf("{\"passed\":true,\"valid_rank_offsets\":%d,\"other_geometry_rejections\":%d,\"CPU_only\":true,\"GPU_math_proof\":false}\n",valid,invalid);}
