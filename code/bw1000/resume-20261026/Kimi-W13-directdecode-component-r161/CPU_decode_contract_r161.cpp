#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>
#include "fp4_direct_pair_r161.hpp"
uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}float val(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
void original(unsigned char p,unsigned char s,float&l,float&h){constexpr float m[]={0,.5,1,1.5,2,3,4,6};auto d=[&](unsigned c){float f=m[c&7];return std::ldexp((c&8)?-f:f,int(s)-127);};l=d(p&15);h=d(p>>4);}
int main(){size_t n=0,normal=0,edge=0;for(int s=0;s<256;++s)for(int p=0;p<256;++p){float a,b,c,d;original(p,s,a,b);fp4DirectPairR161(p,s,c,d,val,original);assert(bits(a)==bits(c)&&bits(b)==bits(d));++n;(s>=2&&s<=252?normal:edge)++;}assert(n==65536);std::cout<<"CPU_FP4_exhaustive_PASS pairs="<<n<<" normal="<<normal<<" edge="<<edge<<" signedzero_subnormal_overflow_exact=1\n";}
