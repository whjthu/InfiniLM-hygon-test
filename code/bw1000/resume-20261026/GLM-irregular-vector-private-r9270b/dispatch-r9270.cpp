#include "glm_vector_dispatch_r8540.hpp"
#include <array>
#include <atomic>
#include <stdexcept>
extern "C" int glm_vec_gate_r8528(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*);
extern "C" int glm_vec_down_r8528(void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*);
extern "C" int glm_grouped_gate_r5123(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*);
extern "C" int glm_grouped_down_r5123(void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*);
namespace {
struct State {std::atomic<unsigned>word{0};std::array<std::atomic<std::uint64_t>,8>counts{};};
std::array<State,8>states;
bool valid(int d){return d>=0&&d<8;}
bool selected(const glm_vector_r8540::Frame&f,std::size_t routes){return f.enabled&&(routes==32||routes==64||routes==128||((states[f.device].word.load(std::memory_order_acquire)&2u)&&(routes==104||routes==120)));}
void record(const glm_vector_r8540::Frame&f,bool gate,bool hit,std::size_t routes,int ret){
 auto&s=states[f.device];s.counts[(gate?0:2)+(hit?0:1)].fetch_add(1,std::memory_order_relaxed);
 if(hit)s.counts[gate?5:6].fetch_add(routes/8,std::memory_order_relaxed);
 if(ret)s.counts[4].fetch_add(1,std::memory_order_relaxed);
}
}
extern "C" int modern_glm_vector_policy_r8540(int d,int mode){
 if(!valid(d)||(mode!=0&&mode!=1))return -1;
 auto&s=states[d];unsigned expected=s.word.load(std::memory_order_acquire);
 for(;;){if(expected&~3u)return -2;unsigned desired=(expected&2u)|unsigned(mode);if(s.word.compare_exchange_weak(expected,desired,std::memory_order_acq_rel)){if((expected&1u)!=unsigned(mode))s.counts[7].fetch_add(1,std::memory_order_relaxed);return 0;}}
}
extern "C" int modern_glm_vector_get_r8540(int d){if(!valid(d))return -1;return int(states[d].word.load(std::memory_order_acquire)&1);}
extern "C" int modern_glm_vector_counts_r8540(int d,std::uint64_t*out,std::size_t n){if(!valid(d)||!out||n!=8)return -1;for(std::size_t j=0;j<8;++j)out[j]=states[d].counts[j].load(std::memory_order_relaxed);return 0;}
extern "C" int modern_glm_irregular_vector_policy_r9270(int d,int mode){
 if(!valid(d)||(mode!=0&&mode!=1))return -1;
 auto&s=states[d];unsigned expected=s.word.load(std::memory_order_acquire);
 for(;;){if(expected&~3u)return -2;unsigned desired=(expected&1u)|(unsigned(mode)<<1);if(s.word.compare_exchange_weak(expected,desired,std::memory_order_acq_rel))return 0;}
}
extern "C" int modern_glm_irregular_vector_get_r9270(int d){if(!valid(d))return -1;return int((states[d].word.load(std::memory_order_acquire)>>1)&1u);}
namespace glm_vector_r8540 {
Frame::Frame(int d):device(d),enabled(false){if(!valid(d))throw std::invalid_argument("vector8540 device");enabled=states[d].word.fetch_add(4,std::memory_order_acq_rel)&1;}
Frame::~Frame(){states[device].word.fetch_sub(4,std::memory_order_release);}
int gate(const Frame&f,void*g,void*u,const void*x,const void*xs,const void*ids,const void*w,const void*ws,const void*up,const void*us,std::size_t routes,std::size_t offset,void*stream){
 bool hit=selected(f,routes);int rc=hit?glm_vec_gate_r8528(g,u,x,xs,ids,w,ws,up,us,routes,offset,stream):glm_grouped_gate_r5123(g,u,x,xs,ids,w,ws,up,us,routes,offset,stream);record(f,true,hit,routes,rc);return rc;
}
int down(const Frame&f,void*out,const void*x,const void*xs,const void*ids,const void*w,const void*ws,std::size_t routes,std::size_t offset,void*stream){
 bool hit=selected(f,routes);int rc=hit?glm_vec_down_r8528(out,x,xs,ids,w,ws,routes,offset,stream):glm_grouped_down_r5123(out,x,xs,ids,w,ws,routes,offset,stream);record(f,false,hit,routes,rc);return rc;
}
}
