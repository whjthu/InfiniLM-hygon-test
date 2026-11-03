#pragma once
#include <cstddef>
#include <stdexcept>
namespace glm_value_r431 {
constexpr size_t heads=8,latent=512,key_nope=192,value=256,head_rows=448,local_rows=3584,global_rows=28672,scale_columns=4;
inline size_t value_row(size_t rank,size_t head,size_t channel){
 if(rank>=8||head>=heads||channel>=value)throw std::invalid_argument("r431 V row bounds");
 return rank*local_rows+head*head_rows+key_nope+channel;
}
inline size_t scale_index(size_t rank,size_t head,size_t channel,size_t column_block){
 if(column_block>=scale_columns)throw std::invalid_argument("r431 scale column");
 return(value_row(rank,head,channel)/128)*scale_columns+column_block;
}
inline size_t diagonal_value_index(size_t row,size_t head,size_t channel){
 if(row>=heads||head>=heads||channel>=value)throw std::invalid_argument("r431 expanded V bounds");
 return(row*heads+head)*value+channel;
}
}
