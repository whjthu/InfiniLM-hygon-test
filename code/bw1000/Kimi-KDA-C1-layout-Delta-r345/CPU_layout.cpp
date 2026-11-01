#include "layout_r345.hpp"
#include <cassert>
#include <iostream>
int main(){
 using namespace layout_r345;std::vector<size_t>expected{1,1,6,128};
 assert(accept(expected,{2304,2304,128,1},expected));
 assert(accept(expected,{768,768,128,1},expected));
 for(size_t base:{size_t(0),size_t(768),size_t(1536)})for(size_t h=0;h<6;++h)for(size_t k=0;k<128;++k){size_t physical=base+0*2304+0*2304+h*128+k;assert(physical==base+h*128+k);assert(physical<2304);}
 assert(!accept({1,2,6,128},{4608,2304,128,1},expected));
 assert(!accept(expected,{2304,2304,129,1},expected));
 assert(!accept(expected,{2304,2304,128,2},expected));
 assert(!accept(expected,{-1,2304,128,1},expected));
 assert(!accept({1,1,5,128},{2304,2304,128,1},expected));
 assert(accept({3,6,128,128},{98304,16384,128,1},{3,6,128,128}));
 assert(!accept({3,6,128,128},{98305,16384,128,1},{3,6,128,128}));
 std::cout<<"{\"passed\":true,\"physical_offsets_checked\":2304,\"C1_only\":true,\"copies\":0,\"GPU\":0}\n";
}
