#include "deep_tail_prefill_scope_r7211.hpp"
#include <cassert>
int main(){using deep_tail_prefill_r7211::matches_metadata;
 assert(matches_metadata(1,512,true));
 assert(!matches_metadata(1,512,false));
 assert(!matches_metadata(1,1,false));assert(!matches_metadata(16,16,false));assert(!matches_metadata(1,16,false));
 assert(!matches_metadata(4,2048,true));assert(!matches_metadata(4,1545,true));assert(!matches_metadata(2,512,true));
 assert(!matches_metadata(1,511,true));assert(!matches_metadata(1,513,true));assert(!matches_metadata(0,512,true));
}
