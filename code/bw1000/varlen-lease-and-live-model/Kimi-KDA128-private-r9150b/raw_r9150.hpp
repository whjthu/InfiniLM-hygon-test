#pragma once
#include <tensor.h>
#include <cstddef>
#include <cstdint>
extern "C" int modern_kimi_kda_threads_raw_r9150(int,void*,void*,size_t,const infini::ops::Tensor*,float,float,bool,int,void*,void*);
extern "C" const char*modern_kimi_kda_threads_error_r9150();
extern "C" int modern_kimi_kda_threads_counts_r9150(int,uint64_t*,size_t);
