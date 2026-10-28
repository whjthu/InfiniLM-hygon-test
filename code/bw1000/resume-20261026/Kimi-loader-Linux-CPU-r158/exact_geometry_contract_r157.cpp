#include "cpu_strided_copy_r2515.hpp"
#include "candidate_rows_r157.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
int main(){
 using Shape=std::vector<std::size_t>;using Strides=std::vector<std::ptrdiff_t>;using Stats=infinilm::runtime::detail::CpuCopyStatsR2515;
 int tests=0;
 for(std::size_t rowbytes:{96u,6u}){
  std::size_t stride=rowbytes*16;std::vector<std::byte>source(3584*stride+128);
  for(std::size_t i=0;i<source.size();++i)source[i]=std::byte((i*53+7)%251);auto before=source;
  for(int rank=0;rank<16;++rank){
   std::vector<std::byte>a(3584*rowbytes+128,std::byte{0x55}),b=a;Stats old,now;
   Shape shape{3584,rowbytes};Strides dst{std::ptrdiff_t(rowbytes),1},src{std::ptrdiff_t(stride),1};
   infinilm::runtime::detail::copy_cpu_strided_r2515(a.data()+64,source.data()+64+rank*rowbytes,shape,dst,src,1,0,old);
   infinilm::runtime::detail::copy_cpu_strided_r157_candidate(b.data()+64,source.data()+64+rank*rowbytes,shape,dst,src,1,0,now);
   if(a!=b||source!=before||old.blocks!=now.blocks||old.block_bytes!=now.block_bytes||old.scalar_leaves!=now.scalar_leaves||old.scalar_bytes!=now.scalar_bytes)throw std::runtime_error("full storage/source/counters differ");++tests;
  }
 }
 std::cout<<"{\"passed\":true,\"exact_W2_TP16_full_storage_source_and_counter_cases\":"<<tests<<"}"<<std::endl;
}
