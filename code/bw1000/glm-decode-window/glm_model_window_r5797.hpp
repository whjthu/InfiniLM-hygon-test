#pragma once
#include "runtime/runtime.hpp"
#include <cstddef>
namespace glm_window_r5795 {
namespace rt=infinilm::runtime;
void prepare(rt::Runtime&);
void arm(rt::Runtime&,int round,int cohort,bool enabled);
const char*read_after_CAPI(rt::Runtime&);
void close_after_drain(rt::Runtime&);
bool mark(rt::Runtime&,const char*);
bool armed()noexcept;
bool request_matches(size_t index,size_t row,size_t past,size_t count)noexcept;
bool enter(rt::Runtime&,size_t layer,int stage,int rank,bool valid);
void finish(rt::Runtime&);
void abort_noexcept()noexcept;
struct Scope {
 rt::Runtime&r;bool active;
 Scope(rt::Runtime&v,size_t layer,int stage,int rank,bool valid):r(v),active(enter(v,layer,stage,rank,valid)){}
 void done(){if(active){finish(r);active=false;}}
 ~Scope(){if(active)abort_noexcept();}
};
}
