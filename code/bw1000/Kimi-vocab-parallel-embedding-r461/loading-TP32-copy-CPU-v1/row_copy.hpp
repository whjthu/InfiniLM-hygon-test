#pragma once
#include "cpu_strided_copy_r2515.hpp"
#include <cstddef>
#include <cstring>
#include <limits>

namespace private_r436 {
// Private CPU-only candidate. Existing caller still owns gather-before-scatter
// and buffer spans. Unsupported geometry executes the exact original helper.
template<std::size_t Fixed>
inline void rows(std::byte *dst,const std::byte *src,std::size_t count,
                 std::ptrdiff_t dstep,std::ptrdiff_t sstep,std::size_t bytes) {
    for(std::size_t r=0;r<count;++r) {
        std::memmove(dst,src,Fixed?Fixed:bytes);
        // Avoid constructing an out-of-allocation pointer after the last row.
        if(r+1<count){dst+=dstep;src+=sstep;}
    }
}

template<class Shape,class Strides>
bool try_row_copy(std::byte *dst,const std::byte *src,const Shape &shape,
                  const Strides &ds,const Strides &ss,std::size_t element,
                  infinilm::runtime::detail::CpuCopyStatsR2515 &stats) {
    if(shape.size()!=2||ds.size()!=2||ss.size()!=2||!element||
       !shape[0]||!shape[1]||ds[1]!=1||ss[1]!=1||ds[0]<0||ss[0]<0)
        return false;
    const auto count=std::size_t(shape[0]),width=std::size_t(shape[1]);
    // Leave fully contiguous, broadcasts, reversals, row overlaps and all
    // overflow/ambiguous cases on the original recursive implementation.
    if(std::size_t(ds[0])<width||std::size_t(ss[0])<width||
       (std::size_t(ds[0])==width&&std::size_t(ss[0])==width))return false;
    const auto max=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(width>max/element||std::size_t(ds[0])>max/element||
       std::size_t(ss[0])>max/element)return false;
    const auto bytes=width*element,dstep=std::size_t(ds[0])*element,
               sstep=std::size_t(ss[0])*element;
    if((count-1)>(max-(bytes-1))/dstep||
       (count-1)>(max-(bytes-1))/sstep||
       count>std::numeric_limits<std::size_t>::max()/bytes)return false;
    switch(bytes) {
        case 6:rows<6>(dst,src,count,dstep,sstep,bytes);break;
        case 96:rows<96>(dst,src,count,dstep,sstep,bytes);break;
        case 16:rows<16>(dst,src,count,dstep,sstep,bytes);break;
        case 32:rows<32>(dst,src,count,dstep,sstep,bytes);break;
        case 64:rows<64>(dst,src,count,dstep,sstep,bytes);break;
        case 128:rows<128>(dst,src,count,dstep,sstep,bytes);break;
        case 256:rows<256>(dst,src,count,dstep,sstep,bytes);break;
        default:rows<0>(dst,src,count,dstep,sstep,bytes);break;
    }
    stats.blocks+=count;stats.block_bytes+=count*bytes;
    return true;
}

template<class Shape,class Strides>
void copy(std::byte *dst,const std::byte *src,const Shape &shape,
          const Strides &ds,const Strides &ss,std::size_t element,
          infinilm::runtime::detail::CpuCopyStatsR2515 &stats) {
    if(!try_row_copy(dst,src,shape,ds,ss,element,stats))
        infinilm::runtime::detail::copy_cpu_strided_r2515(
            dst,src,shape,ds,ss,element,0,stats);
}
}
