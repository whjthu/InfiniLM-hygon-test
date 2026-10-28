#include "loader_registry_clock_r159.hpp"
#include <thread>
#include <vector>
#include <iostream>
#include <stdexcept>
int main(){using namespace loader_registry_r159;std::vector<std::thread>threads;for(int d=0;d<8;++d)threads.emplace_back([d]{for(int i=0;i<3;++i){Clock outer(d,LoadBindingsInclusive);Clock inner(d,StateDictBuild);}try{Clock error(d,CacheFinishInclusive);throw std::runtime_error("synthetic error");}catch(...){};});for(auto&t:threads)t.join();for(int d=0;d<8;++d){std::uint64_t v[8];if(read(d,v,8)||v[0]!=3||v[4]!=1||v[6]!=3)return 1;}std::uint64_t v[8];if(!read(-1,v,8)||!read(0,v,7)||!read(8,v,8))return 2;std::cout<<"{\"passed\":true,\"CPU_only\":true,\"owners\":8,\"exception_scope_recorded\":true,\"clock_is_host_not_GPU\":true}";}
