#include "hc_tail_contract.hpp"
#include "policy.hpp"
extern "C" int hc372_cpu_selected(int mode,const char*flag){try{return hc372::selected(mode,flag)?1:0;}catch(...){return -1;}}
extern "C" int hc372_cpu_cast_square(const uint16_t*bits,float*xf,float*square){
 if(!bits||!xf||!square)return 1;
 for(unsigned i=0;i<16384;++i){float x=hc372::unpack(bits[i]);xf[i]=x;square[i]=x*x;}return 0;
}
extern "C" int hc372_cpu_tail(const float*xf,const float*mixes,const float*base,const float*scale,const float*constants,int iterations,uint16_t*collapsed,float*post,float*comb){
 if(!xf||!mixes||!base||!scale||!constants||!collapsed||!post||!comb||iterations<1||iterations>128)return 1;
 float pre[4];for(unsigned r=0;r<4;++r)hc372::row_weights(r,mixes,base,scale,constants,pre,post,comb);
 for(unsigned i=0;i<4096;++i)collapsed[i]=hc372::collapse_at(xf,pre,i);
 hc372::sinkhorn(comb,constants[1],iterations);return 0;
}
