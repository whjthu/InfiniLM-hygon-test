#include "bindings_cache_r9130.hpp"
#include <unordered_map>
#include <thread>
#include <iostream>
using B=std::unordered_map<int,int>;
int main(){B model{{1,11},{2,22}};loading_r9130::ImmutableBindings<B>c(&model,7);int scans=0;auto build=[&]{++scans;return model;};auto&first=c.get(&model,7,build);if(first.at(1)!=11)return 1;c.get(&model,7,build);if(scans!=1)return 2;bool owner=false;std::thread t([&]{try{c.get(&model,7,build);}catch(...){owner=true;}});t.join();if(!owner)return 3;model[2]=33;try{c.finish(&model,7,build,[](auto&a,auto&b){return a==b;});return 4;}catch(...){}if(c.holds_bindings())return 5;try{c.get(&model,7,build);return 6;}catch(...){}std::cout<<"{\"passed\":true,\"CPU_only\":true,\"checks\":6,\"nativeTensorExecuted\":false,\"benchmarkSpeedClaim\":false}";}
