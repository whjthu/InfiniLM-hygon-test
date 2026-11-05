#include "geometry.hpp"
#include <set>
#include <iostream>
#include <vector>
int main(){using namespace glm_global_tp16_r470;J c;std::cin>>c;
std::set<size_t>heads,rows,columns;for(int rank=0;rank<16;++rank){auto q=geometry(c,rank/8,rank%8,1,32768);if(q["tp_rank"]!=rank||q["local_attention_heads"]!=4||q["local_index_heads"]!=2||q["cache_logical_bytes"]["sum"]!=8355053568ULL||q["Model_READY"]!=false)return 1;
for(auto h:q["head448_interleaved"])heads.insert(h["global_head"].get<size_t>());for(size_t i=rank*1792;i<(rank+1)*1792;++i)rows.insert(i);for(size_t i=rank*1024;i<(rank+1)*1024;++i)columns.insert(i);
if(q["q_b"]["global_offset"]!=rank*1024||q["kv_b"]["global_offset"]!=rank*1792||q["o_proj"]["global_offset"]!=rank*1024)return 2;}
if(heads.size()!=64||rows.size()!=28672||columns.size()!=16384)return 3;
for(auto bad:std::vector<std::pair<int,int>>{{-1,0},{2,0},{0,8}}){try{placement(bad.first,bad.second);return 4;}catch(const std::invalid_argument&){}}
try{geometry(c,0,0,1,1024);return 5;}catch(const std::invalid_argument&){}
std::cout<<J{{"passed",true},{"scope","actual checkpoint config metadata only; no Runtime, weights, fit or SDK proof"},{"ranks",16},{"full_indexer_layers",21},{"GPU",false}}.dump()<<"\n";}
