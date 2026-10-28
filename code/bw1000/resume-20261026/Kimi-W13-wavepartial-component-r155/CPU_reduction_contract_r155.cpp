#include <array>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>
int main(){static_assert(16*4*sizeof(float)==256);std::mt19937 gen(155);std::normal_distribution<float>d;double e2=0,r2=0;for(int trial=0;trial<10000;++trial){std::array<float,256>x;for(auto&v:x)v=d(gen);auto a=x;for(int stride=128;stride;stride/=2)for(int j=0;j<stride;++j)a[j]+=a[j+stride];float partial[4];for(int w=0;w<4;++w){float p[64];for(int l=0;l<64;++l)p[l]=x[w*64+l];for(int stride=32;stride;stride/=2)for(int l=0;l<stride;++l)p[l]+=p[l+stride];partial[w]=p[0];}float b=(partial[0]+partial[2])+(partial[1]+partial[3]);assert(std::isfinite(b));double e=double(a[0])-b;e2+=e*e;r2+=double(a[0])*a[0];}double n=std::sqrt(e2/r2);assert(n<1e-5);std::cout<<"CPU_reduction_contract_PASS NRMSE="<<n<<" scratch_bytes=256 baseline=16384\n";}
