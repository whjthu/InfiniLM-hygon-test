#include "row_copy.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
using Shape=std::vector<std::size_t>;using Strides=std::vector<std::ptrdiff_t>;
using Stats=infinilm::runtime::detail::CpuCopyStatsR2515;
int main(){unsigned cases=0;
 for(auto width:{96u,6u}){const std::size_t n=3584,ss=(width==96?1536:96);
  std::vector<std::byte>src(n*ss),saved;
  for(std::size_t i=0;i<src.size();++i)src[i]=std::byte((i*37+11)%251);
  saved=src;
  for(unsigned rank=0;rank<16;++rank){
   std::vector<std::byte>a(n*width+64,std::byte{0xa5}),b=a;
   Shape shape{n,width};Strides ds{std::ptrdiff_t(width),1},st{std::ptrdiff_t(ss),1};Stats x{},y{};
   infinilm::runtime::detail::copy_cpu_strided_r2515(a.data()+32,src.data()+rank*width,shape,ds,st,1,0,x);
   if(!private_r436::try_row_copy(b.data()+32,src.data()+rank*width,shape,ds,st,1,y))return 2;
   if(a!=b||src!=saved||x.blocks!=n||y.blocks!=n||x.block_bytes!=n*width||y.block_bytes!=n*width||x.scalar_leaves||y.scalar_leaves)return 3;
   ++cases;
  }
 }
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"scope\":\"exact W2 byte geometry, all16 TP offsets; old243 suite not rerun\"}\n";
}
