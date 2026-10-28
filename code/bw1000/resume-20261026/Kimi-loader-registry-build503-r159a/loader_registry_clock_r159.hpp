#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
namespace loader_registry_r159 {
enum Phase {StateDictBuild=0,CacheLookupInclusive=1,CacheFinishInclusive=2,LoadBindingsInclusive=3};
inline std::atomic<std::uint64_t> values[8][8]{};
class Clock final{
 int device_;Phase phase_;std::chrono::steady_clock::time_point start_;
 public:
 Clock(int device,Phase phase):device_(device),phase_(phase),start_(std::chrono::steady_clock::now()){if(device<0||device>=8)throw std::invalid_argument("r159 local owner device");}
 ~Clock()noexcept{auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start_).count();values[device_][2*phase_].fetch_add(1,std::memory_order_relaxed);values[device_][2*phase_+1].fetch_add(std::uint64_t(n),std::memory_order_relaxed);}
 Clock(const Clock&)=delete;Clock&operator=(const Clock&)=delete;
};
inline int read(int device,std::uint64_t*out,std::size_t n){if(device<0||device>=8||!out||n!=8)return 1;for(std::size_t i=0;i<8;++i)out[i]=values[device][i].load(std::memory_order_relaxed);return 0;}
}
