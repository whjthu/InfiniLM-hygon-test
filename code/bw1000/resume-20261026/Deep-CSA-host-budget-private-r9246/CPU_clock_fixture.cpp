#include "deep_csa_host_profile_r9246.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
namespace h=infinilm::models::deepseek_v4::csa_host_r9246;
int main(){ {h::Frame off(0,32,32,true);off.set(h::INDEXER);off.success();}assert(h::samples[0].serial==0);h::mode=1;{h::Frame on(0,32,32,true);on.set(h::INDEXER);on.call(h::DESCRIPTORS,[]{volatile double v=0;for(size_t i=1;i<10000;++i)v+=std::sqrt(double(i));});on.set(h::EXPLICIT_DRAIN);on.success();}const auto&a=h::samples[0];assert(a.serial==1&&a.tokens==32&&a.requests==32&&a.completed==1);double w=0,c=0;for(size_t i=0;i<h::N;++i){assert(a.wall[i]>=0&&a.cpu[i]>=0);w+=a.wall[i];c+=a.cpu[i];}assert(std::abs(w-a.wall[h::N])<1e-7&&std::abs(c-a.cpu[h::N])<1e-7);h::mode=0;{h::Frame off(0,7,7,true);off.set(h::OUTPUT);off.success();}assert(h::samples[0].serial==1);std::cout<<"CLOCK_ONLY_CPU_PASS exclusive13sum/defaultOFF/nestedrestore/serial/publish"<<std::endl;}
