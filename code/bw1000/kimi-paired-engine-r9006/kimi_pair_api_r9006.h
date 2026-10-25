#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Opt in both INFINILM_KIMI_PAIR_TP16_R9006=1 and INFINILM_KIMI_SPEC_R3632=1.
// Math policy must be absent or0. Current TP nodes2,PP2; last arg logical capacity.
void* modern_kimi_engine_create_pair_r9006(const char*,int,int,int,const char*,int,size_t);
const char* modern_kimi_pair_geometry_r9006(size_t);
// out[4] = active,shadow,past,phase. phase1 ready,2 verified,3 replay,4 terminal.
int modern_kimi_pair_rows_r9006(void*,int,int32_t*);
// One logical request, initial prompt only. Standard loading/finish/destroy stay r2475.
int modern_kimi_pair_prefill_r9006(void*,size_t,const int32_t*,const int64_t*,const int64_t*,void*,void*,int*);
// Existing r3632 typed layout: N,count=1,flags=3,read,write,logicalKVrow,past,total,
// offsets[2],ids[N],positions[N],logits[N,V]BF16,IDs[N]I64,publishedFlags.
// In replay phase N must equal retained anchor+accepted prefix and identical IDs.
int modern_kimi_pair_forward_r9006(void*,size_t,size_t,int,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int64_t*,const int64_t*,void*,void*,int*);
// accepted counts proposals only (0..N-1). correction remains next unconsumed anchor.
// continuation0 is caller's terminal EOS/limit decision; this API does not sample EOS.
int modern_kimi_pair_commit_r9006(void*,int,size_t,int);
// Reset all logical requests between prompts; poisoned native-error contexts destroy only.
int modern_kimi_pair_reset_r9006(void*);
#ifdef __cplusplus
}
#endif
