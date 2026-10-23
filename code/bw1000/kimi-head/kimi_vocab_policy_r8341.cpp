#include "kimi_dual_vocab_head_r8341.hpp"
#include <cstdint>
extern "C" int modern_kimi_vocab_head_get_r8341(){try{infinilm::layers::linear::head_policy_r8341::initialize();return infinilm::layers::linear::head_policy_r8341::policy.load();}catch(...){return -1;}}
extern "C" int modern_kimi_vocab_head_set_r8341(int value){
 if(value!=0&&value!=1)return 1;
 try{namespace p=infinilm::layers::linear::head_policy_r8341;p::initialize();std::lock_guard<std::mutex>lock(p::mutex);if(p::in_flight)return 2;p::policy.store(value);return 0;}catch(...){return 3;}
}
extern "C" uint64_t modern_kimi_vocab_head_count_r8341(int device,int field){
 if(device<0||device>=8||field<0||field>2)return UINT64_MAX;
 auto&c=infinilm::layers::linear::head_policy_r8341::counts[device];return field==0?c.baseline.load():field==1?c.sharded.load():c.rows.load();
}
