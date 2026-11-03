#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace deep438 {
inline uint32_t fbits(float f){union{float f;uint32_t u;}v{f};return v.u;}
inline float from_bits(uint32_t u){union{uint32_t u;float f;}v{u};return v.f;}
inline float expand(uint16_t b){return from_bits(uint32_t(b)<<16);}
inline uint16_t bf16(float f){auto u=fbits(f);if((u&0x7fffffffu)>0x7f800000u)return uint16_t((u>>16)|0x40u);return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);}
inline uint16_t host_activation(uint16_t gate,uint16_t up){
 float g=expand(gate),u=expand(up);g=g>10.f?10.f:g;u=u< -10.f? -10.f:u;u=u>10.f?10.f:u;
 g=expand(bf16(g));u=expand(bf16(u));
 // CPU libmath is an independent finite reference; device uses the exact
 // pinned SDK __expf/__frcp_rn/two-__fmul_rn path, not this CPU expf result.
 float sig=1.f/(1.f+expf(-g));float a=g*sig;float v=a*u;return bf16(v);
}
struct HostRow{std::array<uint16_t,256>activation;std::array<int8_t,256>packed;float scale;};
inline HostRow host_row(const uint16_t*gu){
 HostRow out{};std::array<float,256>maxima{};
 for(int i=0;i<256;++i){out.activation[i]=host_activation(gu[i],gu[i+256]);float a=expand(out.activation[i]);if(!std::isfinite(a))throw std::domain_error("Original SDK nonfinite float-to-int conversion is not a defined numeric reference");maxima[i]=fabsf(a);}
 for(int step=128;step;step/=2)for(int i=0;i<step;++i)maxima[i]=fmaxf(maxima[i],maxima[i+step]);
 out.scale=fmaxf(maxima[0]/127.f,1e-8f);float inverse=1.f/out.scale;
 for(int i=0;i<256;++i){float v=expand(out.activation[i])*inverse;int q=v>=0?int(floorf(fabsf(v)+.5f)):-int(floorf(fabsf(v)+.5f));q=q>127?127:(q< -127?-127:q);out.packed[i]=int8_t(q);}
 return out;
}
}
