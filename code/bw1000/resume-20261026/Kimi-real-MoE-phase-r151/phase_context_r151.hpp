#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
namespace moe_phase_r151 {
inline constexpr std::array<const char*,15> names={"MoE.router.begin","MoE.router.end","MoE.down.end","MoE.provider.entry","MoE.clear.end","MoE.pack_routes.end","MoE.build_jobs.end","MoE.fused_W13_dequant_Situ.end","MoE.fused_W2_weighted_scatter.end","MoE.provider.return","MoE.routed_TP_collective.end","MoE.routed_norm.end","MoE.up.end","MoE.shared_including_TP_reduce.end","MoE.add.end"};
template<class Runtime> struct Context {Runtime*r=nullptr;int layer=-1;};
template<class Runtime> inline thread_local Context<Runtime>current;
template<class Runtime,void(*Record)(Runtime&,const char*,int)> void emit(size_t index,void*stream){
 auto c=current<Runtime>;if(!c.r)return;c.r->assertOwnerThread();
 if(index>=names.size()||stream!=c.r->stream())throw std::logic_error("r151 owner stream/index contract");
 Record(*c.r,names[index],c.layer);
}
template<class Runtime,void(*Record)(Runtime&,const char*,int)> class Scope {
 Context<Runtime>prior;
public:
 Scope(Runtime&r,int layer,size_t tokens):prior(current<Runtime>){
  r.assertOwnerThread();current<Runtime> = (tokens==64&&(layer==1||layer==47))?Context<Runtime>{&r,layer}:Context<Runtime>{};
  try{emit<Runtime,Record>(0,r.stream());}catch(...){current<Runtime> = prior;throw;}
 }
 ~Scope()noexcept{current<Runtime> = prior;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
inline constexpr size_t max_original_markers_per_task=47*4+4;
inline constexpr size_t max_extra_markers_per_task=15;
static_assert(3*(max_original_markers_per_task+max_extra_markers_per_task)==621);
static_assert(3*(max_original_markers_per_task+max_extra_markers_per_task)<=640);
}
