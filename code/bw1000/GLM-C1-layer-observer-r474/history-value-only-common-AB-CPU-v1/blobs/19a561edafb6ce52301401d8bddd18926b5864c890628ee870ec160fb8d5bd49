#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <utility>
namespace glm_global_tp16_r470 {
using J=nlohmann::json;
struct Placement {int node,device,global;};
inline Placement placement(int node,int device){if(node<0||node>=2||device<0||device>=8)throw std::invalid_argument("r470 two nodes/eight local devices");return {node,device,node*8+device};}
inline void config(const J&c){
 for(auto p:{std::pair<const char*,int>{"num_hidden_layers",78},{"hidden_size",6144},{"num_attention_heads",64},{"index_n_heads",32},{"q_lora_rank",2048},{"kv_lora_rank",512},{"qk_nope_head_dim",192},{"qk_rope_head_dim",64},{"v_head_dim",256},{"vocab_size",154880}})if(c.at(p.first)!=p.second)throw std::invalid_argument(std::string("r470 checkpoint geometry ")+p.first);
 if(c.at("max_position_embeddings").get<size_t>()<32768||c.at("indexer_types").size()!=78||c.at("mlp_layer_types").size()!=78||c.at("indexer_types").at(0)!="full")throw std::invalid_argument("r470 fullcontext/layerlists");
}
inline J shard(size_t n,size_t k,int dim,int rank,int tp=16){
 if(tp<1||rank<0||rank>=tp||dim<0||dim>1||!n||!k||(dim==0?n:k)%tp)throw std::invalid_argument("r470 weight partition");
 size_t width=(dim==0?n:k)/tp,off=rank*width;
 if(width%128||off%128)throw std::invalid_argument("r470 global128block alignment");
 return J{{"global_shape",{n,k}},{"local_shape",dim==0?J{width,k}:J{n,width}},{"tp_dim",dim},{"global_offset",off},{"global_scale_shape",{(n+127)/128,(k+127)/128}},{"scale_replicated",true},{"row_offset",dim==0?off:0},{"column_offset",dim==1?off:0}};
}
inline J geometry(const J&cfg,int node,int device,size_t rows,size_t capacity){
 config(cfg);auto p=placement(node,device);if(rows!=1||capacity!=32768)throw std::invalid_argument("r470 fixedC1/full32768");
 J heads=J::array();for(int h=0;h<4;++h){size_t global=p.global*4+h,base=global*448;heads.push_back(J{{"global_head",global},{"global_K192_rows",{base,base+192}},{"global_V256_rows",{base+192,base+448}},{"K_scale_blocks",{base/128,(base+191)/128}},{"V_scale_blocks",{(base+192)/128,(base+447)/128}}});}
 size_t full=0;for(const auto&t:cfg.at("indexer_types")){if(t!="full"&&t!="shared")throw std::invalid_argument("r470 indexer kind");full+=t=="full";}
 size_t key=78ULL*capacity*4*256*2,latent=78ULL*capacity*512*2,rotary=78ULL*capacity*64*2,index=full*capacity*128*2;
 return J{{"TP",16},{"PP",1},{"nodes",2},{"local_devices",8},{"node",node},{"device",device},{"tp_rank",p.global},{"world_rank",p.global},{"world_size",16},{"pp_stage",0},{"layer_begin",0},{"layer_end",78},{"rows",rows},{"capacity_per_row",capacity},{"local_attention_heads",4},{"local_index_heads",2},{"root_output_rank",0},{"q_b",shard(16384,2048,0,p.global)},{"kv_b",shard(28672,512,0,p.global)},{"o_proj",shard(6144,16384,1,p.global)},{"head448_interleaved",heads},{"index_head_begin",p.global*2},{"index_head_end",p.global*2+2},{"cache_logical_bytes",{{"projected_K",key},{"latent",latent},{"rotary",rotary},{"full_indexer_K",index},{"sum",key+latent+rotary+index}}},{"full_indexer_layers",full},{"replicated_embedding_BF16_bytes",154880ULL*6144*2},{"lm_head_BF16_local_bytes",9680ULL*6144*2},{"cache_allocated",false},{"weights_loaded",false},{"Model_READY",false},{"fit", "UNKNOWN"}};
}
inline bool requested(const J&c){const char*e=std::getenv("GLM_GLOBAL_TP16_R470");return e&&std::string(e)=="1"&&c.value("glm_global_tp16_r470",false);}
// For future private derived constructors only. No shared or default source includes this.
inline thread_local bool constructing=false;
struct Scope {bool old;explicit Scope(const J&cfg):old(constructing){if(!requested(cfg))throw std::invalid_argument("r470 explicit JSON+env optin required");constructing=true;}~Scope(){constructing=old;}Scope(const Scope&)=delete;};
template<class Rank>inline bool admitted(const Rank&r){return constructing&&r.tp_size==16&&r.pp_size==1&&r.pp_stage==0&&r.world_size==16&&r.tp_rank>=0&&r.tp_rank<16&&r.world_rank==r.tp_rank&&r.device.index()==r.tp_rank%8&&r.comm&&r.world_comm;}
}
