#include "deep_csa_bank_plan_r9249.hpp"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
namespace b=infinilm::models::deepseek_v4::bank_copy_r9249;
int main(){std::array<unsigned char,16*1024>src{},dst{},reference{};for(size_t i=0;i<src.size();++i)src[i]=i%251;auto p=b::plan(3,2,16,16,7,reinterpret_cast<uintptr_t>(dst.data()),reinterpret_cast<uintptr_t>(src.data()));assert(p.valid&&p.bytes==7168&&p.source_offset==2048&&p.destination_offset==3072);std::memcpy(dst.data()+p.destination_offset,src.data()+p.source_offset,p.bytes);for(size_t row=0;row<7;++row)std::memcpy(reference.data()+(3+row)*1024,src.data()+(2+row)*1024,1024);assert(dst==reference);assert(!b::plan(3,2,16,16,0,100000,200000).valid);assert(!b::plan(3,2,4,16,2,100000,200000).valid);assert(!b::plan(0,15,16,16,2,100000,200000).valid);assert(!b::plan(0,0,16,16,1,100000,100000).valid);assert(!b::plan(1,0,16,16,1,UINTPTR_MAX-512,200000).valid);assert(!b::plan(0,0,SIZE_MAX,16,1,100000,200000).valid);std::cout<<"BANK_PLAN_CPU_PASS rowoffset/nonoverlap/overflow/OOB/readonly/canary/numeric"<<std::endl;}
