#pragma once
#include <cstddef>
#include <stdexcept>
namespace kimi_scale_exact_r201 {
struct Geometry {bool long_factory;std::size_t tp,pp,experts,hidden,local_intermediate;bool hygon,eager;};
inline bool enabled(const Geometry&g){
 if(!g.long_factory)return false;
 if(g.tp!=16||g.pp!=2||g.experts!=896||g.hidden!=3584||g.local_intermediate!=192||!g.hygon||!g.eager)
  throw std::invalid_argument("r201 exact scales require scoped32K TP16PP2 I192");
 return true;
}
struct Ticket {const void*runtime;const void*object;};
// Constructor-stack admission only: no Tensor/Runtime ownership, registry,
// pointer reuse cache or cross-generation persistence. Cleared on return/throw.
inline thread_local const Ticket*active=nullptr;
class Scope final {
 Ticket ticket_;bool armed_=false;
public:
 Scope(const Geometry&g,const void*runtime,const void*object):ticket_{runtime,object}{
  if(!enabled(g))return;
  if(!runtime||!object||active)throw std::invalid_argument("r201 nested/invalid constructor admission");
  active=&ticket_;armed_=true;
 }
 ~Scope()noexcept{if(armed_)active=nullptr;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
inline bool selected(const void*runtime,const void*object){
 if(!active)return false;
 if(active->runtime!=runtime||active->object!=object)throw std::invalid_argument("r201 foreign constructor admission");
 return true;
}
}
