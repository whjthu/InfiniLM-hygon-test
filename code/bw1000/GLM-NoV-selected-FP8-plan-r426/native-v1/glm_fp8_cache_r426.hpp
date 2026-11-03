#pragma once
#include <cstddef>
#include <cstdint>
extern "C" const char*glm_fp8_cache_error_r426();
extern "C" void*glm_fp8_cache_create_r426(int,void*,const void*);
extern "C" int glm_fp8_cache_run_r426(void*,const void*,void*,int,int,size_t,size_t,size_t,size_t,size_t,size_t,size_t,void*,const void*,const void*,const void*,const void*);
extern "C" int glm_fp8_cache_stats_r426(void*,uint64_t*,size_t);
extern "C" int glm_fp8_cache_clear_r426(void*,const void*,void*);
extern "C" int glm_fp8_cache_destroy_r426(void*,const void*,void*);

#include "geometry_r426.hpp"
