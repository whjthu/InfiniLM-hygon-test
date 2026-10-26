#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace infinilm::models::deepseek_v4::bank_copy_r9249 {
struct Plan{size_t destination_offset=0,source_offset=0,bytes=0;bool valid=false;};
// Pure-HOST range/overflow/nonoverlap proof for contiguous BF16 rows of width512.
inline Plan plan(size_t dr,size_t sr,size_t drows,size_t srows,size_t count,uintptr_t da,uintptr_t sa){
 Plan p;constexpr size_t rowbytes=512*2;const auto max=std::numeric_limits<size_t>::max();const auto ptrmax=std::numeric_limits<uintptr_t>::max();
 if(!count||dr>drows||sr>srows||count>drows-dr||count>srows-sr||drows>max/rowbytes||srows>max/rowbytes)return p;
 p.destination_offset=dr*rowbytes;p.source_offset=sr*rowbytes;p.bytes=count*rowbytes;
 if(da>ptrmax-p.destination_offset||sa>ptrmax-p.source_offset)return p;auto d=da+p.destination_offset,s=sa+p.source_offset;
 if(d>ptrmax-p.bytes||s>ptrmax-p.bytes||!(d+p.bytes<=s||s+p.bytes<=d))return p;p.valid=true;return p;
}
}
