#include "row_copy.hpp"
#include <chrono>
#include <iostream>
#include <vector>
using Shape=std::vector<std::size_t>;
using Strides=std::vector<std::ptrdiff_t>;
using Stats=infinilm::runtime::detail::CpuCopyStatsR2515;
using Clock=std::chrono::steady_clock;
volatile unsigned long long observed=0;
int main(){
    std::cout<<"{\"cases\":[";
    bool comma=false;
    for(auto bytes:{32u,64u,128u}){
        const std::size_t n=32768,w=bytes/2;
        Shape sh{n,w};Strides ds{std::ptrdiff_t(w),1},ss{std::ptrdiff_t(w*2),1};
        std::vector<std::byte>src(n*bytes*2),old(n*bytes),out(n*bytes);
        for(std::size_t i=0;i<src.size();++i)src[i]=std::byte((i*37+11)%251);
        Stats a{},b{};
        infinilm::runtime::detail::copy_cpu_strided_r2515(old.data(),src.data(),sh,ds,ss,2,0,a);
        private_r436::copy(out.data(),src.data(),sh,ds,ss,2,b);
        if(old!=out||a.blocks!=n||b.blocks!=n||a.block_bytes!=b.block_bytes)return 2;
        if(comma)std::cout<<',';
        comma=true;
        std::cout<<"{\"block_bytes\":"<<bytes<<",\"rows\":"<<n
                 <<",\"source_row_stride_bytes\":"<<bytes*2
                 <<",\"logical_copy_bytes_per_call\":"<<n*bytes<<",\"schedule\":[";
        const int schedule[]={0,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0};
        for(unsigned i=0;i<sizeof(schedule)/sizeof(schedule[0]);++i){
            auto begin=Clock::now();Stats s{};
            for(unsigned rep=0;rep<32;++rep){
                if(schedule[i])private_r436::copy(out.data(),src.data(),sh,ds,ss,2,s);
                else infinilm::runtime::detail::copy_cpu_strided_r2515(out.data(),src.data(),sh,ds,ss,2,0,s);
                observed+=std::to_integer<unsigned>(out[(rep*239)%out.size()]);
            }
            auto seconds=std::chrono::duration<double>(Clock::now()-begin).count();
            if(s.blocks!=n*32||s.block_bytes!=n*bytes*32||out!=old)return 3;
            if(i)std::cout<<',';
            std::cout<<"{\"mode\":"<<schedule[i]<<",\"calls\":32,\"seconds\":"<<seconds<<'}';
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"passed\":true,\"checksum\":"<<observed
             <<",\"scope\":\"synthetic CPU strided-row copies only; actual model shape distribution and loading speedup unknown\"}\n";
}
