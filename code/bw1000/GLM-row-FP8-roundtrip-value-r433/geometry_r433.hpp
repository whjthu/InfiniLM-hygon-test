#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace glm_value_r433 {
constexpr size_t heads=8,latent=512,value=256,head_rows=448,local_rows=3584,scale_columns=4;
constexpr uint8_t fp8_one=0x38,fp8_zero=0;
inline size_t value_row(size_t rank,size_t head,size_t channel){
 if(rank>=8||head>=heads||channel>=value)throw std::invalid_argument("r433 V row bounds");
 return rank*local_rows+head*head_rows+192+channel;
}
inline size_t scale_index(size_t rank,size_t head,size_t channel,size_t column_block){
 if(column_block>=scale_columns)throw std::invalid_argument("r433 scale column");
 return(value_row(rank,head,channel)/128)*scale_columns+column_block;
}
inline size_t value_column(size_t head){return value_row(0,head,0);}
inline size_t row_offset(size_t rank){if(rank>=8)throw std::invalid_argument("r433 TP8 rank bounds");return rank*local_rows;}
inline uint8_t identity_code(size_t row,size_t column){
 if(row>=latent||column>=latent)throw std::invalid_argument("r433 identity bounds");
 return row==column?fp8_one:fp8_zero;
}
constexpr size_t value_weight_bytes=heads*latent*value*4;
constexpr size_t query_identity_FLOPs=2ull*2048*latent*latent;
constexpr size_t setup_identity_FLOPs=2ull*latent*local_rows*latent;
}
