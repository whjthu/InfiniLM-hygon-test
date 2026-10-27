#include "deep_prefix_device_scope_r135.hpp"
#include <cassert>
#include <iostream>
struct Mock {static inline int device=0,sets=0,fail=0;static int get(int*p){*p=device;return 0;}static int set(int d){++sets;if(fail)return 1;device=d;return 0;}};
using Guard=infinilm::models::deepseek_v4::prefix_r135::ScopedDevice<Mock>;
int main(){
 {Guard g(3);assert(Mock::device==3&&g.previous()==0);{Guard nested(3);assert(Mock::device==3);}assert(Mock::device==3);}assert(Mock::device==0);
 try{Guard g(7);assert(Mock::device==7);throw std::logic_error("unwind");}catch(const std::logic_error&){}assert(Mock::device==0);
 Mock::fail=1;try{Guard bad(2);assert(false);}catch(const std::runtime_error&){}assert(Mock::device==0);Mock::fail=0;
 try{Guard bad(-1);assert(false);}catch(const std::runtime_error&){}assert(Mock::device==0);
 std::cout<<"{\"passed\":true,\"CPU_mock_only\":true,\"GPU_qualified\":false,\"cases\":4}";
}
