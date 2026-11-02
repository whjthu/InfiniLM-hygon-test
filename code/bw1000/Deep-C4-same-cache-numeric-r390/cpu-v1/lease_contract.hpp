#pragma once
#include <cstdint>
#include <stdexcept>
namespace same_cache_r390 {
// This HOST-only protocol is also used by the actual adapter. The producer's
// private Pending owner/revision remains authoritative and is checked by commit.
struct Lease {
 uint64_t generation=0;unsigned scored=0;bool active=false,committed=false;
 void idle()const{if(active)throw std::logic_error("Live Pending forbids mutation/reset");}
 uint64_t begin(){idle();++generation;scored=0;active=true;committed=false;return generation;}
 void check(uint64_t token)const{if(!active||token!=generation)throw std::logic_error("Stale/foreign Pending lease");}
 void score(uint64_t token,int mode){check(token);if(mode<0||mode>1||(scored&(1u<<mode)))throw std::logic_error("Invalid/repeated score arm");scored|=1u<<mode;}
 void finish(uint64_t token){check(token);if(scored!=3)throw std::logic_error("Both same-cache score arms required");active=false;committed=true;}
 void readable(uint64_t token)const{if(token!=generation||(!active&&!committed))throw std::logic_error("Released lease proof");}
 void reset(){idle();++generation;scored=0;committed=false;}
};
}
