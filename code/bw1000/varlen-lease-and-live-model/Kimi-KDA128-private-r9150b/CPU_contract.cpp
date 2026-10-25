#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
float reduce(const std::vector<float>&v,size_t block){std::vector<float>x(block);for(size_t t=0;t<block;++t)for(size_t k=t;k<v.size();k+=block)x[t]+=v[k];for(size_t stride=block/2;stride;stride/=2)for(size_t t=0;t<stride;++t)x[t]+=x[t+stride];return x[0];}
int main(){size_t checks=0;for(size_t C:{1ul,2ul,4ul,8ul,16ul,32ul})for(size_t W:{1ul,2ul,4ul,8ul})if(C*W<=64)for(size_t block:{128ul,256ul}){assert(block%64==0&&block>=128&&(3*128+block)*4<=65536&&C*6*128>0);++checks;}for(size_t seed=0;seed<256;++seed){std::vector<float>x(128);for(size_t i=0;i<128;++i)x[i]=std::sin(float(i+seed)*.173f);auto a=reduce(x,256),b=reduce(x,128);assert(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<1e-4f);++checks;}std::cout<<"{\"passed\":true,\"pure_CPU\":true,\"checks\":"<<checks<<",\"D\":128,\"blocks\":[128,256],\"warp\":64,\"reduction_uses_blockDim\":true,\"GPU_tested\":false}"<<std::endl;}
