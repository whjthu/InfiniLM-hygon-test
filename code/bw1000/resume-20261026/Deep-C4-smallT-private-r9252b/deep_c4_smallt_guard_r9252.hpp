#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
namespace infinilm::models::deepseek_v4::small_c4_r9252 {
template<class Requests>bool decode(size_t tokens,const Requests&requests){return tokens>=1&&tokens<=7&&requests.size()==tokens&&std::all_of(requests.begin(),requests.end(),[](const auto&q){return q.tokens==1&&q.past>0&&q.past<std::numeric_limits<size_t>::max();});}
template<class Requests>bool eligible(bool on,size_t tokens,const Requests&requests,size_t topk){return on&&decode(tokens,requests)&&std::all_of(requests.begin(),requests.end(),[&](const auto&q){return(q.past+1)/4<=topk;});}
}
