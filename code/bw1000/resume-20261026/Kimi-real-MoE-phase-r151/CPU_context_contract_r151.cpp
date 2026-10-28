#include "phase_context_r151.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
struct Runtime{int id;void assertOwnerThread(){}void*stream(){return reinterpret_cast<void*>(size_t(id));}};
std::vector<int>layers;bool fail=false;
void record(Runtime&,const char*,int layer){if(fail)throw std::runtime_error("injected record error");layers.push_back(layer);}
using Scope=moe_phase_r151::Scope<Runtime,record>;
int main(){Runtime a{1},b{2};assert(!moe_phase_r151::current<Runtime>.r);
 {Scope s(a,1,64);assert(layers.size()==1);assert(moe_phase_r151::current<Runtime>.r==&a);{Scope n(b,47,64);assert(moe_phase_r151::current<Runtime>.r==&b);}assert(moe_phase_r151::current<Runtime>.r==&a);{Scope n(b,48,64);moe_phase_r151::emit<Runtime,record>(99,nullptr);assert(!moe_phase_r151::current<Runtime>.r);}assert(moe_phase_r151::current<Runtime>.r==&a);bool caught=false;try{moe_phase_r151::emit<Runtime,record>(3,b.stream());}catch(...){caught=true;}assert(caught);}
 assert(!moe_phase_r151::current<Runtime>.r);size_t n=layers.size();{Scope s(a,1,8);}assert(layers.size()==n);fail=true;try{Scope s(a,47,64);assert(false);}catch(...){}assert(!moe_phase_r151::current<Runtime>.r);fail=false;
 try{Scope s(a,1,64);throw std::runtime_error("forward error");}catch(...){}assert(!moe_phase_r151::current<Runtime>.r);
 std::cout<<"{\"passed\":true,\"CPU_only\":true,\"GPU\":false,\"Native\":false,\"cases\":8,\"max_markers_3calls\":621}\n";
}
