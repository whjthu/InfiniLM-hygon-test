#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace kimi_ar_storage_r456 {
constexpr size_t context=32768,logical=1,physical=1,pool=2,pages=128,blocks=132,page_size=256;
inline void cache_geometry(size_t rows,size_t count,size_t width){
 if(rows!=physical||count!=blocks||width!=page_size)throw std::invalid_argument("r456 AR C1 exact cache: row0 reserved, row1 active, pages1..128 plus guard blocks");
}
inline void state_geometry(size_t length_rows,bool spec){if(length_rows!=pool||spec)throw std::invalid_argument("r456 no shadow/speculative state");}
inline void topology(bool enabled,bool long_scope,size_t tp,size_t pp,size_t local_i){
 if(enabled&&(!long_scope||tp!=32||pp!=1||local_i!=96))throw std::invalid_argument("r456 explicit AR TP32PP1 I96 only");
}
// Constructor stack only. Distinct from the retained scale ticket and cleared
// on normal/exceptional return; no pointer registry or added model fields.
struct Ticket{const void*runtime;const void*object;};
inline thread_local const Ticket*active=nullptr;
class Scope final {
 Ticket ticket_;bool armed_=false;
public:
 Scope(bool enabled,bool long_scope,size_t tp,size_t pp,size_t local_i,const void*r,const void*o):ticket_{r,o}{
  topology(enabled,long_scope,tp,pp,local_i);if(!enabled)return;
  if(!r||!o||active)throw std::invalid_argument("r456 foreign/nested constructor");active=&ticket_;armed_=true;
 }
 ~Scope()noexcept{if(armed_)active=nullptr;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
inline bool selected(const void*r,const void*o){if(!active)return false;if(active->runtime!=r||active->object!=o)throw std::invalid_argument("r456 foreign W2 allocation");return true;}
// Used by the actual freeze_input acceptance boundary and pure CPU references.
inline void request(size_t row,size_t write,size_t lengths,size_t page_width,bool spec){
 state_geometry(lengths,spec);if(row!=1||write!=1||page_width!=128)throw std::invalid_argument("r456 only existing logical row1 read/write");
}
}
