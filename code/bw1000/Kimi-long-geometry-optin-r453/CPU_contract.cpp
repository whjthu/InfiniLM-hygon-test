// Actual private pure admission + actual head expected-row registration. This
// executable links no Native/SDK and constructs no Runtime/Tensor/Model.
#include "long_geometry_r453.hpp"
#include "scale_exact_scope_r201.hpp"
#include "kimi_shard_stats_r9183.hpp"
#include "reference_pipeline_partition.hpp"
#include <iostream>
#include <set>
#include <vector>
#include <type_traits>
namespace g=kimi_long_geometry_r453;
static size_t checks=0;
static void require(bool v){if(!v)throw std::logic_error("CPU geometry contract");++checks;}
template<class F>static void reject(F f){try{f();}catch(const std::invalid_argument&){++checks;return;}throw std::logic_error("missing reject");}
int main(){try{
 struct LegacyGeometry {bool long_factory;size_t tp,pp,experts,hidden,local_intermediate;bool hygon,eager;};
 static_assert(sizeof(kimi_scale_exact_r201::Geometry)==sizeof(LegacyGeometry),"retained geometry descriptor layout");
 for(auto t:std::array<std::pair<size_t,size_t>,3>{{{16,2},{8,4},{32,1}}}){
  const size_t tp=t.first,pp=t.second;const bool opt=tp!=16;const auto a=g::geometry(opt,tp,pp);
  require(a.heads*tp==96&&a.expert_i*tp==3072&&a.shared_i*tp==6144&&a.head_rows*tp==163840);
  require(tp*pp==32&&a.tp_nodes*pp==4&&a.head_bytes==a.head_rows*7168*2);
  std::set<size_t>world;std::vector<size_t>layercoverage(93,0);size_t emitter_nodes=0;
  for(size_t stage=0;stage<pp;++stage){const auto l=g::layers(pp,stage);
   require(l==infinilm::models::kimi_k3::kimi_k3_pipeline_layer_range(93,pp,stage));
   for(size_t i=l.first;i<l.second;++i)++layercoverage.at(i);
  }
  for(size_t n=0;n<4;++n){if(n==(pp-1)*a.tp_nodes)++emitter_nodes;
   for(size_t d=0;d<8;++d){const auto r=g::rank(opt,tp,pp,n,d);require(world.insert(r.world_rank).second);
    require(r.world_rank==r.stage*tp+r.tp_rank&&r.tp_rank==(n%a.tp_nodes)*8+d);
    require(r.head_end-r.head_begin==a.head_rows&&r.head_end<=163840);
    require(r.embedding==(r.stage==0)&&r.head==(r.stage+1==pp)&&r.emitter_node==(n==(pp-1)*a.tp_nodes));
    require(a.expert_i%32==0&&7168%32==0&&3584%32==0);
   }
  }
  require(world.size()==32&&emitter_nodes==1);for(auto n:layercoverage)require(n==1);
  // Actual inline scale admission and lifetime. No Runtime pointer is dereferenced.
  kimi_scale_exact_r201::Geometry s{true,tp,pp,896,3584,a.expert_i,true,true};
  require(kimi_scale_exact_r201::enabled(s,opt));int runtime=0,object=0;
  {kimi_scale_exact_r201::Scope scope(s,&runtime,&object,opt);require(kimi_scale_exact_r201::selected(&runtime,&object));}
  require(kimi_scale_exact_r201::active==nullptr);
  auto bad=s;bad.local_intermediate+=32;reject([&]{kimi_scale_exact_r201::enabled(bad,opt);});
  bad=s;bad.eager=false;reject([&]{kimi_scale_exact_r201::enabled(bad,opt);});
  bad=s;bad.hygon=false;reject([&]{kimi_scale_exact_r201::enabled(bad,opt);});
  if(opt)reject([&]{kimi_scale_exact_r201::enabled(s);});
  // Actual stats header's registration, duplicate and offset rejection.
  using namespace kimi_shard_r9183;auto h=head_geometry_r410(tp);
  require(h.rows==a.head_rows&&h.bytes==a.head_bytes&&h.samples==std::array<size_t,3>{0,h.rows/2,h.rows-1});
  json ranks=json::array();for(size_t rank=0;rank<tp;++rank)ranks.push_back({{"rank",rank},{"row_begin",rank*h.rows},{"digests",{rank,rank+1,rank+2}}});
  expected_ready=false;set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});require(expected_ready&&expected_tp==tp);
  try{set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});return 10;}catch(const std::logic_error&){++checks;}
  expected_ready=false;ranks[0]["row_begin"]=1;reject([&]{set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});});
  ranks[0]["row_begin"]=0;ranks[tp-1]["rank"]=0;reject([&]{set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});});
  reject([&]{g::rank(opt,tp,pp,4,0);});reject([&]{g::rank(opt,tp,pp,0,8);});
 }
 for(size_t tp:{size_t(0),size_t(4),size_t(48)})reject([&]{g::head_geometry(tp);});
 reject([]{g::geometry(false,8,4);});reject([]{g::geometry(false,32,1);});reject([]{g::geometry(true,16,2);});reject([]{g::geometry(true,8,2);});
 for(int p:{0,512,32256})g::call_scope(0,1,1,p,512,3);
 g::call_scope(1,1,1,32767,1,3);checks+=4;
 reject([]{g::call_scope(0,1,1,32257,512,3);});reject([]{g::call_scope(0,1,1,0,513,3);});
 reject([]{g::call_scope(1,1,1,0,1,3);});reject([]{g::call_scope(1,1,1,10,2,3);});
 reject([]{g::call_scope(1,2,1,10,1,3);});reject([]{g::call_scope(1,1,2,10,1,3);});
 reject([]{g::call_scope(1,1,1,-1,1,3);});reject([]{g::call_scope(2,1,1,1,1,3);});reject([]{g::call_scope(1,1,1,1,1,7);});
 require(g::context==32768&&g::physical==2&&g::pool==3&&g::blocks==260&&g::page_size==256);
 std::cout<<nlohmann::json({{"passed",true},{"checks",checks},{"profiles",{{16,2},{8,4},{32,1}}},{"all_world_ranks",32},{"layers_each_profile",93},{"actual_scale_and_head_header",true},{"Runtime_created",false},{"Native_dlopen",false},{"device_calls",0}}).dump()<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
