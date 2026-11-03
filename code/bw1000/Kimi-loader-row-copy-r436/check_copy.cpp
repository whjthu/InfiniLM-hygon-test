#include "row_copy.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using Shape=std::vector<std::size_t>;
using Strides=std::vector<std::ptrdiff_t>;
using Stats=infinilm::runtime::detail::CpuCopyStatsR2515;
static void check(const Shape &shape,const Strides &ds,const Strides &ss,
                  std::size_t es,std::size_t dbase=4096,std::size_t sbase=4096) {
    std::vector<std::byte>src(32768),a(32768,std::byte{0xa5}),b=a;
    for(std::size_t i=0;i<src.size();++i)src[i]=std::byte((i*37+11)%251);
    Stats x{},y{};
    infinilm::runtime::detail::copy_cpu_strided_r2515(
        a.data()+dbase,src.data()+sbase,shape,ds,ss,es,0,x);
    private_r436::copy(b.data()+dbase,src.data()+sbase,shape,ds,ss,es,y);
    if(a!=b||x.blocks!=y.blocks||x.block_bytes!=y.block_bytes||
       x.scalar_leaves!=y.scalar_leaves||x.scalar_bytes!=y.scalar_bytes)
        throw std::runtime_error("Original bytes/guards/stats mismatch");
}
int main(){
    std::size_t count=0;
    for(auto es:{1u,2u,4u})for(auto n:{1u,3u,16u})
      for(auto w:{1u,8u,16u,32u,64u,128u})
       for(auto gaps:std::array<std::array<int,2>,4>{{{0,0},{0,3},{7,0},{3,7}}}){
        check({n,w},{std::ptrdiff_t(w+gaps[0]),1},
                    {std::ptrdiff_t(w+gaps[1]),1},es);++count;
       }
    for(auto es:{1u,2u,4u}){
      check({3,7},{7,1},{0,1},es);++count; // broadcast fallback
      check({3,7},{7,1},{-11,1},es);++count; // reverse outer fallback
      check({3,7},{7,1},{11,-1},es);++count; // reverse inner fallback
      check({3,7},{1,3},{7,1},es);++count; // transposed fallback
      check({2,3,7},{21,7,1},{35,9,1},es);++count;
      check({0,7},{7,1},{9,1},es);++count;
      check({3,0},{7,1},{9,1},es);++count;
      check({}, {}, {},es);++count;
    }
    // The helper itself preserves per-row memmove semantics. Cross-row overlap
    // retains the caller's mandatory gather path; this is a single-row probe.
    for(auto es:{1u,2u,4u}){
      std::vector<std::byte>a(4096),b;
      for(std::size_t i=0;i<a.size();++i)a[i]=std::byte(i%251);
      b=a;
      Stats x{},y{};Shape sh{1,32};Strides ds{40,1},ss{48,1};
      infinilm::runtime::detail::copy_cpu_strided_r2515(a.data()+2,a.data(),sh,ds,ss,es,0,x);
      private_r436::copy(b.data()+2,b.data(),sh,ds,ss,es,y);
      if(a!=b)throw std::runtime_error("Single-row overlap mismatch");
      ++count;
    }
    std::cout<<"{\"passed\":true,\"cases\":"<<count
             <<",\"scope\":\"local CPU geometry only; no target performance\"}\n";
}
