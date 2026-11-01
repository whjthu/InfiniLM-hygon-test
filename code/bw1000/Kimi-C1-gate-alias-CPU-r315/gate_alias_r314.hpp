#pragma once
#include "runtime/runtime.hpp"
#include <cstdint>
namespace kimi_gate_alias_r314 {
namespace rt=infinilm::runtime;
void mode(rt::Runtime&,bool);
bool enabled(rt::Runtime&);
void hit(rt::Runtime&,bool);
void stats(rt::Runtime&,std::uint64_t*,std::size_t);
void clear(rt::Runtime&);
}
#ifdef KIMI_GATE_ALIAS_IMPL_R314
#include <unordered_map>
#include <array>
#include <stdexcept>
namespace kimi_gate_alias_r314 {
struct State{bool on=false;std::array<std::uint64_t,3>counts{};};
static thread_local std::unordered_map<rt::Runtime*,State>states;
void mode(rt::Runtime&r,bool on){r.assertOwnerThread();states[&r].on=on;}
bool enabled(rt::Runtime&r){r.assertOwnerThread();auto it=states.find(&r);return it!=states.end()&&it->second.on;}
void hit(rt::Runtime&r,bool selected){r.assertOwnerThread();auto it=states.find(&r);if(it==states.end())return;auto&s=it->second;++s.counts[0];++s.counts[selected?1:2];}
void stats(rt::Runtime&r,std::uint64_t*out,std::size_t n){r.assertOwnerThread();if(!out||n!=3)throw std::invalid_argument("314 stats size");auto it=states.find(&r);for(std::size_t i=0;i<n;++i)out[i]=it==states.end()?0:it->second.counts[i];}
void clear(rt::Runtime&r){r.assertOwnerThread();states.erase(&r);}
}
#endif
