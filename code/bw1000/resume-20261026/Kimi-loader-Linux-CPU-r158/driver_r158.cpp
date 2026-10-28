#include "benchmark_api_r158.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
using Copy=decltype(&baseline_r158);
static std::atomic<std::uint64_t> calls{0},blocks{0},bytes{0},leaves{0},scalar{0};
static void add(const CountersR158&s){calls.fetch_add(1,std::memory_order_relaxed);blocks.fetch_add(s.blocks,std::memory_order_relaxed);bytes.fetch_add(s.bytes,std::memory_order_relaxed);leaves.fetch_add(s.leaves,std::memory_order_relaxed);scalar.fetch_add(s.scalar,std::memory_order_relaxed);}
struct Bank{std::vector<std::byte>w,s;Bank(int n):w(3584*1536),s(3584*96){for(std::size_t i=0;i<w.size();++i)w[i]=std::byte((i*73+19+n)%251);for(std::size_t i=0;i<s.size();++i)s[i]=std::byte((i*41+7+n)%251);}};
int main(int argc,char**argv){
 bool smoke=argc>1&&std::strcmp(argv[1],"--smoke")==0;
 ShapeR158 ws{3584,96},ss{3584,6};StridesR158 wd{96,1},ww{1536,1},sd{6,1},sw{96,1};
 std::cout<<"{\"CPU_only\":true,\"separate_TUs\":true,\"LTO\":false,\"shared_global_source\":true,\"rank_ids\":[0,1,2,3,4,5,6,7],\"cold_claim\":false,\"trials\":[";bool first=true;
 for(int workload=0;workload<2;++workload){
  const int nbank=workload?(smoke?4:64):1,iterations=smoke?128:(workload?512:4096);
  std::vector<Bank>banks;banks.reserve(nbank);for(int i=0;i<nbank;++i)banks.emplace_back(i);
  std::array<std::vector<std::byte>,8>dw,ds;for(int r=0;r<8;++r){dw[r].resize(3584*96+64,std::byte{0x55});ds[r].resize(3584*6+64,std::byte{0x55});}
  for(int trial=0;trial<4;++trial){
   Copy fn=(trial==0||trial==3)?baseline_r158:candidate_r158;calls=0;blocks=0;bytes=0;leaves=0;scalar=0;
   std::atomic<int>ready{0};std::atomic<bool>go{false};std::vector<std::thread>threads;
   for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{ready.fetch_add(1);while(!go.load(std::memory_order_acquire))std::this_thread::yield();
    for(int i=0;i<iterations;++i){const auto&b=banks[std::size_t(i)%banks.size()];CountersR158 a,c;
     fn(dw[rank].data(),b.w.data()+rank*96,ws,wd,ww,&a);add(a);
     fn(ds[rank].data(),b.s.data()+rank*6,ss,sd,sw,&c);add(c);}});
   while(ready.load()!=8)std::this_thread::yield();auto start=std::chrono::steady_clock::now();go.store(true,std::memory_order_release);for(auto&t:threads)t.join();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   const auto&last=banks[std::size_t(iterations-1)%banks.size()];
   for(int rank=0;rank<8;++rank){for(int row=0;row<3584;++row){if(std::memcmp(dw[rank].data()+row*96,last.w.data()+row*1536+rank*96,96)||std::memcmp(ds[rank].data()+row*6,last.s.data()+row*96+rank*6,6))throw std::runtime_error("output mismatch");}
    if(!std::all_of(dw[rank].end()-64,dw[rank].end(),[](std::byte x){return x==std::byte{0x55};})||!std::all_of(ds[rank].end()-64,ds[rank].end(),[](std::byte x){return x==std::byte{0x55};}))throw std::runtime_error("guard mismatch");}
   if(calls!=std::uint64_t(8*iterations*2)||blocks!=std::uint64_t(8*iterations*2)*3584||bytes!=std::uint64_t(8*iterations)*3584*102||leaves||scalar)throw std::runtime_error("counter mismatch");
   if(!first)std::cout<<",";first=false;std::cout<<"{\"workload\":\""<<(workload?"rotating_initialized_banks_not_cold":"warm_shared_one_bank")<<"\",\"trial\":"<<trial<<",\"mode\":\""<<((trial==0||trial==3)?"baseline":"candidate")<<"\",\"source_banks\":"<<nbank<<",\"source_working_set_bytes\":"<<std::uint64_t(nbank)*3584*1632<<",\"iterations_per_owner\":"<<iterations<<",\"seconds\":"<<seconds<<",\"top_level_calls\":"<<calls<<",\"bulk_blocks\":"<<blocks<<",\"bulk_bytes\":"<<bytes<<",\"byte_and_guard_and_counter_pass\":true}";
  }
 }
 std::cout<<"],\"Model_or_H2D_performance_proven\":false}"<<std::endl;
}
