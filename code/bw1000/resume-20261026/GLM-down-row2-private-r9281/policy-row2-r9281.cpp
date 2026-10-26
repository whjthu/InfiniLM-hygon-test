#include <atomic>
#include <cstddef>
#include <cstdint>
extern "C" int glm_vec_down_reference_r9281(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_vec_down_row2_r9281(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
namespace { std::atomic<unsigned> word{0};std::atomic<uint64_t> counts[4]; }
// PRIVATE singleGPUfixture. Caller must drain stream before changing policy.
extern "C" int modern_glm_down_row2_policy_r9281(int d,int mode){if(d!=0||mode<0||mode>1)return -1;auto x=word.load();do{if(x&~1u)return-2;}while(!word.compare_exchange_weak(x,unsigned(mode)));if(x!=unsigned(mode))++counts[3];return 0;}
extern "C" int modern_glm_down_row2_get_r9281(int d){return d? -1:int(word.load()&1u);}
extern "C" int modern_glm_down_row2_counts_r9281(int d,uint64_t*out,size_t n){if(d||!out||n!=4)return-1;for(int i=0;i<4;++i)out[i]=counts[i].load();return 0;}
extern "C" int glm_vec_down_r8528(void*out,const void*x,const void*xs,const void*ids,const void*w,const void*s,size_t routes,size_t offset,void*stream){
 if(!out||!x||!xs||!ids||!w||!s||!routes||routes>128||routes%8||offset>1792||offset%256){++counts[2];return-10;}
 unsigned state=word.fetch_add(2);int mode=state&1u;++counts[mode];int rc=mode?glm_vec_down_row2_r9281(out,x,xs,ids,w,s,routes,offset,stream):glm_vec_down_reference_r9281(out,x,xs,ids,w,s,routes,offset,stream);word.fetch_sub(2);if(rc)++counts[2];return rc;
}
