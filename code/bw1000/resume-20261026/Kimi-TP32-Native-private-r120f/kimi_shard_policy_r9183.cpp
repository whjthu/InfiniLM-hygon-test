#include "kimi_shard_stats_r9183.hpp"
#include <cstdint>
namespace{thread_local std::string serialized,error;}
extern "C" int modern_kimi_vocab_head_get_r8341(){return -1;}
extern "C" int modern_kimi_vocab_head_set_r8341(int){return 1;}
extern "C" uint64_t modern_kimi_vocab_head_count_r8341(int,int){return UINT64_MAX;}
extern "C" const char*modern_kimi_shard_head_info_r9183(){try{serialized=kimi_shard_r9183::info().dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int modern_kimi_shard_head_expected_r9183(const char*p){try{kimi_shard_r9183::set_expected(nlohmann::json::parse(p));return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_kimi_shard_head_error_r9183(){return error.c_str();}
