#pragma once
#include <cstddef>
extern "C" int kimi_split_fa_pack(void*,void*,void*,const void*,const void*,int,std::size_t,std::size_t,void*);
extern "C" int kimi_split_fa_factors(void*,void*,const void*,std::size_t,int,void*);
extern "C" int kimi_split_fa_merge(void*,const void*,const void*,std::size_t,int,void*);
extern "C" int kimi_split_fa_finish(void*,const void*,const void*,std::size_t,std::size_t,void*);
extern "C" int kimi_split_fa_lengths(void*,std::size_t,std::size_t,std::size_t,std::size_t,void*);
