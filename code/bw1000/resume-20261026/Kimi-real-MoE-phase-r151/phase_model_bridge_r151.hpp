#pragma once
#include "phase_context_r151.hpp"
#include "kimi_model_events_r8669zq.hpp"
namespace moe_phase_r151 {
using R=infinilm::runtime::Runtime;
inline void record(R&r,const char*name,int layer){kimi_model_events_r8669zq::mark(r,name,layer,false,true);}
using OwnerScope=Scope<R,record>;
inline void mark(size_t index){auto c=current<R>;if(c.r)emit<R,record>(index,c.r->stream());}
}
extern "C" int modern_kimi_moe_phase_mark_r151(unsigned index,void*stream)noexcept{try{moe_phase_r151::emit<moe_phase_r151::R,moe_phase_r151::record>(index,stream);return 0;}catch(...){return 1;}}
