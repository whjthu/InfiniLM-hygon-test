// Pure local C++ host-byte test. No project runtime, CUDA, weight or device headers.
#include "input_guard.hpp"
#include <iostream>
#include <functional>
#include <string>
using namespace c4_input_guard_r483;
static size_t rejected=0,verified=0;
static void require(bool b){if(!b)throw std::runtime_error("CPU assertion");}
static void reject(const std::function<void()>&f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught);++rejected;}
int main(){try{
 for(size_t item:std::vector<size_t>{1,2,4,8}){
  std::vector<uint8_t>p(17*item);for(size_t i=0;i<p.size();++i)p[i]=uint8_t(i*31+7);
  auto g=layout(p.size(),std::vector<size_t>{1,17},item);require(g.prefix_elements*item==256&&g.total_elements*item==p.size()+512);
  Buffer b(p.data(),g);b.actual=b.expected;b.verify();++verified;
  for(size_t at:std::vector<size_t>{0,255,256,256+p.size()-1,256+p.size(),g.total_bytes-1}){
   b.actual=b.expected;b.actual[at]^=1;reject([&]{b.verify();});
  }
  b.actual=b.expected;b.actual.pop_back();reject([&]{b.verify();});
  b.actual=b.expected;b.actual.push_back(0);reject([&]{b.verify();});
 }
 reject([]{layout(0,std::vector<size_t>{0},2);});
 reject([]{layout(16,std::vector<size_t>{8},3);});
 reject([]{layout(16,std::vector<size_t>{},2);});
 reject([]{layout(16,std::vector<long long>{-8},2);});
 reject([]{layout(16,std::vector<size_t>{0,8},2);});
 reject([]{layout(15,std::vector<size_t>{8},2);});
 reject([]{layout(16,std::vector<size_t>{std::numeric_limits<size_t>::max(),2},2);});
 reject([]{layout(16,std::vector<size_t>{std::numeric_limits<size_t>::max()/2+1},2);});
 reject([]{layout(std::numeric_limits<size_t>::max(),std::vector<size_t>{std::numeric_limits<size_t>::max()},1);});
 reject([]{auto g=layout(16,std::vector<size_t>{8},2);Buffer b(nullptr,g);});
 // Largest current checkpoint input and one full4096-row hidden input without copying any checkpoint values.
 for(size_t n:std::vector<size_t>{8388608,4096*4096*2}){
  std::vector<uint8_t>p(n,0x31);Buffer b(p.data(),layout(n,std::vector<size_t>{n/2},2));b.actual=b.expected;b.verify();++verified;
 }
 require(rejected==42&&verified==6);
 std::cout<<"PASS host_guard_cases="<<verified<<" rejected_corrupt_or_invalid="<<rejected<<" Native=0 SDK=0 GPU=0\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
