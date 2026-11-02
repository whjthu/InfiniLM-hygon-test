#pragma once
#include <cstddef>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa::v_on_demand_r400 {
constexpr size_t max_block_rows=8192;
template<class Fn>inline void for_blocks(size_t rows,Fn fn){
 if(!rows||rows>32768)throw std::invalid_argument("r400 C1 selected union outside1..32768");
 for(size_t begin=0;begin<rows;begin+=max_block_rows){size_t count=rows-begin;if(count>max_block_rows)count=max_block_rows;fn(begin,count);}
}
}
