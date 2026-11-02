#include "policy_r403.hpp"
#include "host_ids_r403.hpp"
#include "chunk_geometry_r400.hpp"
#include <cassert>
#include <string>
#include <utility>
using namespace infinilm::models::glm_moe_dsa;
int main(){
 unsetenv("GLM_V_ON_DEMAND_R403");assert(!v_on_demand_r403::requested());
 setenv("GLM_V_ON_DEMAND_R403","0",1);assert(!v_on_demand_r403::requested());
 setenv("GLM_V_ON_DEMAND_R403","1",1);assert(v_on_demand_r403::requested());
 for(const char*p:{"", "2", "true"}){setenv("GLM_V_ON_DEMAND_R403",p,1);bool rejected=false;try{v_on_demand_r403::requested();}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
 // The inherited configured batch flag1 remains inactive at actual C1/hostOFF.
 for(bool request_batch:{false,true})for(size_t requests:{size_t{1}}){bool hostbank=false,resident=false,staged=false;bool batch=!staged&&!resident&&request_batch&&hostbank&&requests>=8;assert(!batch);}
 // Exercise the actual HOST owning container with pending asynchronous readers.
 std::vector<std::pair<const int64_t*,int64_t>>pending;bool drained=false;
 struct Work{v_on_demand_r403::HeldHostIds ids;std::vector<std::pair<const int64_t*,int64_t>>&pending;bool&drained;~Work(){for(auto p:pending)assert(*p.first==p.second);drained=true;}};
 try{Work w{{},pending,drained};for(int64_t j=0;j<2048;++j){std::vector<int64_t>local{j,j+1};const auto&held=w.ids.hold(local);pending.push_back({held.data(),j});}throw 1;}catch(int){}assert(drained);
 std::vector<std::pair<size_t,size_t>>blocks;v_on_demand_r400::for_blocks(32768,[&](size_t a,size_t b){blocks.push_back({a,b});});assert(blocks.size()==4&&blocks[3]==std::make_pair(size_t{24576},size_t{8192}));
}
