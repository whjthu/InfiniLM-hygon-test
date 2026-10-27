#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
namespace owner_contract_r134 {
inline void mask(const std::vector<int>&actual,const std::vector<int>&allowed,size_t width){
 if(actual.size()!=width||width==0)throw std::runtime_error("r134 actual Runtime owner width");
 auto sorted=actual;std::sort(sorted.begin(),sorted.end());if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())throw std::runtime_error("r134 duplicate CPU");
 for(int c:actual)if(std::find(allowed.begin(),allowed.end(),c)==allowed.end())throw std::runtime_error("r134 owner CPU outside original Slurm allowed");
}
inline void all(const std::array<std::vector<int>,8>&masks,const std::vector<int>&allowed){
 std::vector<int>seen;for(const auto&m:masks){mask(m,allowed,4);for(int c:m){if(std::find(seen.begin(),seen.end(),c)!=seen.end())throw std::runtime_error("r134 Runtime owner masks overlap");seen.push_back(c);}}
}
}
