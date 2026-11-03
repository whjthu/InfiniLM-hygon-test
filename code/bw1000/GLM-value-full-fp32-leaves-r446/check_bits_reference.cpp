// Prepared CPU contract only. Does not execute/load Runtime or device code.
#include "fp32_bits_r446.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <utility>
static float from_bits(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
static uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
int main(){
 using namespace glm_value_r446;
 size_t finite=0;
 for(uint32_t b=0;b<65536;++b){
  const auto u=bf16_to_f32_bits(uint16_t(b));
  if(finite_bits(u)){assert(f32_to_bf16_bits(u)==b);assert(bits(from_bits(u))==u);++finite;}
 }
 assert(finite==65280);
 assert(f32_to_bf16_bits(0x3f808000u)==0x3f80); // even lower
 assert(f32_to_bf16_bits(0x3f818000u)==0x3f82); // odd lower
 assert(f32_to_bf16_bits(0xbf808000u)==0xbf80);
 assert(f32_to_bf16_bits(0x00008000u)==0x0000); // BF16 subnormal midpoint
 assert(f32_to_bf16_bits(0x00018000u)==0x0002);
 assert(f32_to_bf16_bits(0x80000000u)==0x8000);
 assert(f32_to_bf16_bits(0x7f7fffffu)==0x7f80); // overflow must flag device status
 assert(!finite_bits(0x7f800000u)&&!finite_bits(0x7fc00000u));
 assert((f32_to_bf16_bits(0x7f800001u)&0x7fffu)==0x7fc0u);
 // Both real geometries, fixed float-only fma reference. Identity/one-hot and
 // signed cancellation are algebra cases, never Model activation qualification.
 for(const auto&shape:{std::pair<size_t,size_t>{2048,512},{512,256}}){
  const size_t k=shape.first,n=shape.second;
  std::vector<float>x(k),w(k*n),y(n);
  for(size_t j=0;j<k;++j)x[j]=(j%2)?-1.f:1.f;
  for(size_t j=0;j<k;++j)for(size_t c=0;c<n;++c)w[j*n+c]=j==c?0.5f:0.f;
  for(size_t c=0;c<n;++c){float z=0.f;for(size_t j=0;j<k;++j)z=std::fma(x[j],w[j*n+c],z);y[c]=z;assert(y[c]==x[c]*0.5f);}
 }
 std::cout<<"{\"passed\":true,\"finite_BF16_encodings\":"<<finite<<",\"GPU\":false,\"Runtime\":false,\"Model\":false}"<<std::endl;
}
