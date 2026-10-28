#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace kimi_long_agent_r179 {
inline constexpr size_t context=32768,logical=1,physical=2,pool=3,pages=128,blocks=260,block_size=256;
inline void validate_cache(size_t rows,size_t num_blocks,size_t page_size){
 if(rows!=physical||num_blocks!=blocks||page_size!=block_size)throw std::invalid_argument("r179 exact scoped cache geometry required");
}
// Shared by the ACTUAL freeze_input path and pure CPU malformed-table tests.
// Outer freeze_input already validated all vector/tensor dimensions and owner.
inline void validate_row_table(size_t kv_row,int32_t past,int32_t total,int32_t begin,int32_t end,size_t page_width,const int32_t*table,const int64_t*slots,const int64_t*positions){
 if(kv_row!=1||past<0||total<=past||total>int32_t(context)||begin<0||end<=begin||int64_t(end)-begin!=int64_t(total)-past||page_width!=pages||!table||!slots||!positions)throw std::invalid_argument("r179 row/past/width/context geometry");
 for(size_t j=0;j<pages;++j)if(table[j]!=int32_t(j+1))throw std::invalid_argument("r179 logical1 must own pages1..128");
 for(size_t j=0;j<size_t(end-begin);++j){int64_t position=int64_t(past)+int64_t(j);const int64_t slot=(1+position/256)*256+position%256;if(positions[j]!=position||slots[j]!=slot)throw std::invalid_argument("r179 absolute position/cache slot geometry");}
}
}
