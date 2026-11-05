#pragma once
#include <cstddef>
#include <cstdint>
extern "C" int kimi_split_store(void*,const void*,const void*,const void*,int,std::size_t,const std::int64_t*,const std::int64_t*,void*);
extern "C" int kimi_split_gather(void*,const void*,const void*,int,std::size_t,std::size_t,std::size_t,std::size_t,void*);
extern "C" int kimi_split_mask(void*,std::size_t,std::size_t,std::size_t,std::size_t,void*);
