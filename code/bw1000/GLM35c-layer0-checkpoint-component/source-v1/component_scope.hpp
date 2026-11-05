#pragma once
#include "glm_moe_dsa_projected_cache_r22.hpp"
#include "glm_registry_geometry_r184b.hpp"
#include <thread>
#include <cstring>
namespace infinilm::models::glm_moe_dsa::layer0_component {
namespace rt=runtime;
// Explicit constructor-local capability. No environment switch or Model default.
inline thread_local rt::Runtime* constructing=nullptr;
struct Construction final {
 rt::Runtime&r;
 explicit Construction(rt::Runtime&owner):r(owner){r.assertOwnerThread();if(constructing)throw std::logic_error("nested Layer0 component construction");constructing=&r;}
 ~Construction(){constructing=nullptr;}
};
inline void member(rt::Runtime&r,const engine::distributed::RankInfo&rank,size_t layer,size_t rows,size_t capacity,size_t topk){
 if(constructing==&r){
  r.assertOwnerThread();if(layer!=0||rows!=1||capacity!=32768||topk!=2048||rank.tp_size!=8||rank.world_size!=8||rank.pp_size!=1||rank.pp_stage!=0||rank.tp_rank<0||rank.tp_rank>=8||rank.world_rank!=rank.tp_rank||rank.device!=r.device()||!rank.comm||rank.world_comm!=rank.comm)throw std::invalid_argument("Layer0 component requires genuine TP8/world8/PP1/C1/full32K");
 }else registry_geometry_r184b::member(rank.tp_size,rank.pp_size,rank.pp_stage,layer);
}
struct Field {std::string name;rt::Tensor tensor;rt::Shape shape;rt::DataType dtype;std::vector<unsigned char>bytes;};
struct Capture;
inline thread_local Capture* active=nullptr;
struct Capture final {
 rt::Runtime&r;std::vector<Field>fields;size_t retained=0;bool complete=false;
 static constexpr size_t limit=64u*1024u*1024u;
 explicit Capture(rt::Runtime&owner):r(owner){r.assertOwnerThread();if(active)throw std::logic_error("nested component capture");active=this;}
 Capture(const Capture&)=delete;Capture&operator=(const Capture&)=delete;
 ~Capture(){active=nullptr;try{r.syncStream();}catch(...){std::terminate();}}
 void keep(const char*name,const rt::Tensor&t){
  if(!t||t->device()!=r.device()||!t->is_contiguous()||complete)throw std::invalid_argument("component snapshot tensor");
  for(const auto&f:fields)if(f.name==name)throw std::logic_error("duplicate component snapshot");
  if(t->nbytes()>limit-retained)throw std::length_error("component snapshot 64MiB owner limit");
  retained+=t->nbytes();fields.push_back({name,t,t->shape(),t->dtype(),{}});
 }
 // Invoked only after the original attention forward has returned/drained.
 // Each D2H remains on the owner thread. CPU getters never own a Runtime/Tensor.
 void finish(){r.assertOwnerThread();r.syncStream();for(auto&f:fields){f.bytes.resize(f.tensor->nbytes());r.memcpyD2H(f.bytes.data(),f.tensor->data(),f.bytes.size());}r.syncStream();for(auto&f:fields)f.tensor.reset();complete=true;active=nullptr;}
};
inline void keep(const char*name,const rt::Tensor&t){if(active)active->keep(name,t);}
}
