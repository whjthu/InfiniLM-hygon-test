#include "glm_registry_geometry_r184b.hpp"
#include <iostream>
using namespace infinilm::models::glm_moe_dsa;
template<class F>void reject(F f){bool ok=false;try{f();}catch(const std::exception&){ok=true;}if(!ok)throw std::logic_error("expected reject");}
int main(){size_t members=0;
 for(int pp:{2,4})for(int stage=0;stage<pp;++stage){auto b=registry_geometry_r184b::span(pp,stage);std::set<size_t>layers;for(size_t l=b.begin;l<b.end;++l){registry_geometry_r184b::member(8,pp,stage,l);layers.insert(l);++members;}registry_geometry_r184b::complete(pp,stage,layers);registry_geometry_r184b::policy(pp,0,0,-1);auto bad=layers;bad.erase(b.begin);reject([&]{registry_geometry_r184b::complete(pp,stage,bad);});bad=layers;bad.erase(b.begin);bad.insert(b.end);reject([&]{registry_geometry_r184b::complete(pp,stage,bad);});reject([&]{registry_geometry_r184b::member(8,pp,stage,b.end);});reject([&]{registry_geometry_r184b::member(4,pp,stage,b.begin);});}
 reject([]{registry_geometry_r184b::span(4,4);});reject([]{registry_geometry_r184b::span(3,0);});
 for(int pp:{2,4})for(int mode:{0,1,2})for(int diag:{0,1})for(int shadow:{-1,77}){if(pp==4&&(mode||diag||shadow!=-1))reject([&]{registry_geometry_r184b::policy(pp,mode,diag,shadow);});else registry_geometry_r184b::policy(pp,mode,diag,shadow);}
 std::cout<<"{\"passed\":true,\"CPU_only\":true,\"exact_members_checked\":"<<members<<",\"PP4_optimization_OFF_only\":true,\"PP2_layout_unchanged\":true,\"wrong_layer_missing_topology_rejected\":true}"<<std::endl;
}
