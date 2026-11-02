#pragma once
#include <cstdint>
#include <stdexcept>
namespace infinilm::pd_r382 {
// Shared by the actual CAPI and host CPU boundary checks. No Runtime or mock
// Native owner is constructed by these host-only predicates.
inline void poison_generation(bool &failed,uint64_t current,uint64_t expected){
    if(!expected||expected!=current)throw std::invalid_argument("PD poison stale generation");
    failed=true;
}
inline void retire_session(bool &failed,bool importing){if(importing)failed=true;}
inline void require_regular(bool failed,bool active){
    if(failed||active)throw std::invalid_argument("PD session active or context failed");
}
}
