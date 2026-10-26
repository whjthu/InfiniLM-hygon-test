#pragma once
#include <algorithm>
#include "glm_model_resident_policy_r5640.hpp"
namespace glm_model_resident_r5640 {
inline Slots propose32_r9183(const Slots& original,const std::vector<Request>& requests,size_t maximum){
 original.owner();if(requests.empty()||requests.size()>(maximum==32?32:16)||(maximum!=16&&maximum!=32))throw std::invalid_argument("9183 request count/maxactive");
 std::array<bool,logical_rows>seen{};
 for(auto q:requests){if(q.row>=logical_rows||seen[q.row]||!q.tokens||q.past>capacity||q.tokens>capacity-q.past)throw std::invalid_argument("9183 wholelist row/past/tokens");seen[q.row]=true;}
 Slots pending=original;
 for(size_t begin=0;begin<requests.size();begin+=16){std::vector<Request>part(requests.begin()+begin,requests.begin()+std::min(requests.size(),begin+16));pending=pending.propose(part,maximum);}
 return pending;
}
}
