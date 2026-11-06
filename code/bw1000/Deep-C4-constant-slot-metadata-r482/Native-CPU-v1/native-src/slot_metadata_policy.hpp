#pragma once
#include <array>
#include <cstddef>
#include <atomic>
#include <cstdint>
namespace infinilm::models::deepseek_v4::slot_metadata_r482 {
// Startup fixed before any PagedIndexerCache; no request-time mode changes.
inline std::atomic<int> policy{0};
inline std::array<std::array<std::atomic<uint64_t>,5>,8> counts{};
inline int set(int mode) {
 if(mode!=0&&mode!=1)return 1;
 int old=policy.load(std::memory_order_acquire);
 while(old<2)if(policy.compare_exchange_weak(old,mode,std::memory_order_acq_rel))return 0;
 return 2;
}
inline bool freeze(){return(policy.fetch_or(2,std::memory_order_acq_rel)&1)!=0;}
inline void count(int device,std::size_t field,uint64_t value=1){counts.at(std::size_t(device)).at(field).fetch_add(value,std::memory_order_relaxed);}
inline uint64_t read(int device,int field){if(device<0||device>=8||field<0||field>=5)return UINT64_MAX;return counts[std::size_t(device)][std::size_t(field)].load(std::memory_order_relaxed);}
}
