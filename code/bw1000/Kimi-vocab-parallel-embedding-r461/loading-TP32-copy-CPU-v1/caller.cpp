#include "row_copy.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using Shape=std::vector<std::size_t>;
using Strides=std::vector<std::ptrdiff_t>;
using Stats=infinilm::runtime::detail::CpuCopyStatsR2515;
using Clock=std::chrono::steady_clock;
volatile unsigned long long observed=0;
void require(bool yes){if(!yes)throw std::runtime_error("TP32 copy contract");}
bool guards(const std::vector<std::byte>&v,std::size_t body){return std::all_of(v.begin(),v.begin()+64,[](auto x){return x==std::byte{0x6d};})&&std::all_of(v.begin()+64+body,v.end(),[](auto x){return x==std::byte{0x6d};});}
int main(){
 std::cout<<std::setprecision(17)<<"{\"geometries\":[";bool comma=false;
 for(std::size_t w:{48u,3u}){
  const std::size_t rows=3584,stride=w*32,body=rows*w;
  const Shape shape{rows,w};const Strides ds{std::ptrdiff_t(w),1},ss{std::ptrdiff_t(stride),1};
  std::vector<std::byte>src(rows*stride+128,std::byte{0x6d}),baseline(body+128,std::byte{0x6d}),candidate(body+128,std::byte{0x6d});
  for(std::size_t i=0;i<rows*stride;++i)src[64+i]=std::byte((i*37+11)%251);
  const auto original=src;
  for(std::size_t rank=0;rank<32;++rank){
   Stats a{},b{};const auto*source=src.data()+64+rank*w;
   infinilm::runtime::detail::copy_cpu_strided_r2515(baseline.data()+64,source,shape,ds,ss,1,0,a);
   require(private_r436::try_row_copy(candidate.data()+64,source,shape,ds,ss,1,b));
   require(a.blocks==rows&&a.block_bytes==body&&a.scalar_leaves==0&&a.scalar_bytes==0);
   require(a.blocks==b.blocks&&a.block_bytes==b.block_bytes&&baseline==candidate&&guards(candidate,body)&&guards(baseline,body)&&src==original);
   for(std::size_t r=0;r<rows;++r)require(std::equal(candidate.begin()+64+r*w,candidate.begin()+64+(r+1)*w,source+r*stride));
  }
  if(comma){std::cout<<',';}
  comma=true;
  std::cout<<"{\"width\":"<<w<<",\"rows\":"<<rows<<",\"stride\":"<<stride<<",\"all32offsets_byte_canary_PASS\":true,\"arms\":[";
  const std::array<int,6>schedule{{0,1,0,1,1,0}};
  for(std::size_t arm=0;arm<schedule.size();++arm){
   Stats stats{};const auto started=Clock::now();
   for(unsigned pass=0;pass<8;++pass)for(std::size_t rank=0;rank<32;++rank){
    const auto*source=src.data()+64+rank*w;
    if(schedule[arm])private_r436::copy(candidate.data()+64,source,shape,ds,ss,1,stats);
    else infinilm::runtime::detail::copy_cpu_strided_r2515(candidate.data()+64,source,shape,ds,ss,1,0,stats);
    observed+=std::to_integer<unsigned>(candidate[64+(pass*239)%body]);
   }
   const auto seconds=std::chrono::duration<double>(Clock::now()-started).count();
   require(stats.blocks==rows*256&&stats.block_bytes==body*256&&stats.scalar_leaves==0&&guards(candidate,body)&&src==original);
   if(arm){std::cout<<',';}
   std::cout<<"{\"mode\":"<<schedule[arm]<<",\"warm\":"<<(arm<2?"true":"false")<<",\"calls\":256,\"seconds\":"<<seconds<<",\"blocks\":"<<stats.blocks<<",\"bytes\":"<<stats.block_bytes<<'}';
  }
  std::cout<<"]}";
 }
 // One required rejection/fallback for contiguous source/destination at new width3.
 Shape sh{2,3};Strides st{3,1};std::array<std::byte,6>a{},b{};Stats rejected{},fallback{};
 require(!private_r436::try_row_copy(a.data(),b.data(),sh,st,st,1,rejected));
 private_r436::copy(a.data(),b.data(),sh,st,st,1,fallback);require(a==b&&fallback.blocks==1&&fallback.block_bytes==6);
 std::cout<<"],\"fallback_PASS\":true,\"passed\":true,\"checksum\":"<<observed<<",\"scope\":\"synthetic warm CPU RAM; no weights/SDK/Native/H2D/GPU/Model gain\"}\n";
}
