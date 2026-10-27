// Future bounded qualification only: no checkpoint, no model forward, no full93
// parameter construction. One H3 KDA + one H3 MLA + shard head + actual93 caches.
#include "tp32_contract_r120f.hpp"
#include "kimi_shard_head_r9183.hpp"
#include "config/config_factory.hpp"
#include "models/kimi_k3/kimi_k3_delta_attention.hpp"
#include "models/kimi_k3/kimi_k3_mla_attention.hpp"
#include "models/kimi_k3/kimi_k3_allocate_cache.hpp"
#include "engine/distributed/communication_group.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <array>
#include <thread>
#include <memory>
#include <vector>
#include <set>
#include <cstdio>
namespace {using J=nlohmann::json;namespace rt=infinilm::runtime;namespace km=infinilm::models::kimi_k3;namespace en=infinilm::engine::distributed;thread_local std::string result;
J memory(){size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess)throw std::runtime_error("r120f cudaMemGetInfo");if(free<=kimi_tp32_r120f::FloorBytes)throw std::runtime_error("r120f unchanged4GiB floor");return{{"free",free},{"total",total},{"used",total-free}};}
J weights(infinilm::weights::ModelWeights&m){size_t bytes=0;J shapes=J::object();for(auto&x:m.state_dict()){auto t=x.second.tensor();if(t){bytes+=t->nbytes();shapes[x.first]={{"shape",t->shape()},{"bytes",t->nbytes()}};}}return{{"logical_bytes",bytes},{"tensors",shapes},{"payload_loaded",false}};}
void transport(rt::Runtime&r,const en::RankInfo&i){
 auto local=rt::Tensor::empty(r,{1,5120},rt::DataType::kBFloat16),out=rt::Tensor::empty(r,{32,5120},rt::DataType::kBFloat16);std::vector<uint16_t>src(5120,uint16_t(0x3f80+i.tp_rank)),dst(32*5120);r.memcpyH2D(local->data(),src.data(),src.size()*2,false);en::allgather_(r,out,local,i.comm);r.syncStream();r.memcpyD2H(dst.data(),out->data(),dst.size()*2);for(size_t rank=0;rank<32;++rank)for(size_t col=0;col<5120;++col)if(dst[rank*5120+col]!=uint16_t(0x3f80+rank))throw std::runtime_error("r120f real32 BF16 allgather ownership");
}
}
extern "C" const char*modern_kimi_TP32_construct_result_r120f(){return result.c_str();}
extern "C" int modern_kimi_TP32_construct_r120f(const char*payload){J out;try{
 kimi_tp32_r120f::enabled_environment();auto q=J::parse(payload);int node=q.at("node_rank");if(node<0||node>=4)throw std::invalid_argument("r120f node");auto raw=q.at("config");raw["infinilm_text_only"]=true;raw["infinilm_kimi_global_tp_adapter"]=true;raw["text_config"]["infinilm_kimi_global_tp_adapter"]=true;raw["infinilm_kimi_engine_adapter"]=true;raw["infinilm_kimi_compact_cache_r4974"]=false;raw["text_config"]["infinilm_kimi_compact_cache_r4974"]=false;raw["dtype"]="bfloat16";raw["text_config"]["dtype"]="bfloat16";raw["quantization_config"]=nullptr;raw.erase("lm_head_output_dtype");raw["text_config"].erase("lm_head_output_dtype");auto cfg=infinilm::config::ConfigFactory::createConfig(raw.dump());
 if(cfg->get<size_t>("num_hidden_layers")!=93||cfg->get<size_t>("num_attention_heads")!=96||cfg->get<size_t>("head_dim")!=192||cfg->get_config_json().at("linear_attn_config").at("num_heads")!=96)throw std::invalid_argument("r120f original MLA/KDA global heads");
 en::DistConfig topo(8);topo.pp_size=1;topo.pp_stage=0;topo.tp_node_size=4;topo.tp_node_rank=node;topo.master_addr=q.at("master_addr");topo.master_port=q.at("master_port");std::array<J,8>owners;std::vector<std::thread>threads;
 {en::CommunicationGroup group(topo,rt::Device::Type::kHygon);
  for(int local=0;local<8;++local)threads.emplace_back([&,local]{try{
   auto info=group.get_rank_info(local);kimi_tp32_r120f::topology(node,local,info.tp_size,info.tp_rank,info.pp_size,info.pp_stage,info.world_size,info.world_rank);if(info.device.type()!=rt::Device::Type::kHygon||info.device.index()!=local||!info.comm)throw std::runtime_error("r120f actual local device/comm");infini::rt::DeviceGuard guard{info.device.type(),info.device.index(),true};auto owner=rt::Runtime::create(info.device);auto&r=*owner;r.enableOwnerDrainFailStop();J row={{"global_rank",info.tp_rank},{"local_rank",local},{"before_components",memory()},{"checkpoint_opened",false},{"model_forward_calls",0}};
   if(row["before_components"]["free"].get<uint64_t>()<=kimi_tp32_r120f::FloorBytes+3ULL*1024*1024*1024)throw std::runtime_error("r120f bounded component constructor3GiB reservation");
   {auto kda=std::make_unique<km::KimiK3DeltaAttention>(r,cfg,0,info);auto mla=std::make_unique<km::KimiK3MLAAttention>(r,cfg,3,info,infinilm::backends::AttentionBackend::PAGED_ATTN);auto head=std::make_unique<infinilm::layers::linear::ShardOnlyVocabHeadR9183>(r,7168,163840,rt::DataType::kBFloat16,std::nullopt,info);
    row["KDA"]=weights(*kda);row["MLA"]=weights(*mla);row["head"]=weights(*head);if(head->weight()->shape()!=rt::Shape{5120,7168}||head->weight()->nbytes()!=73400320)throw std::runtime_error("r120f head ctor shard");
    auto kw=kda->state_dict();for(const char*n:{"q_proj.weight","k_proj.weight","v_proj.weight"})if(kw.at(n).tensor()->shape()!=rt::Shape{384,7168})throw std::runtime_error("r120f KDA actualQKV384");if(kw.at("o_proj.weight").tensor()->shape()!=rt::Shape{7168,384})throw std::runtime_error("r120f KDA actual o_proj384");
    auto mw=mla->state_dict();if(mw.at("q_b_proj.weight").tensor()->shape().at(0)!=576||mw.at("kv_b_proj.weight").tensor()->shape()!=rt::Shape{768,512}||mw.at("g_proj.weight").tensor()->shape()!=rt::Shape{384,7168}||mw.at("o_proj.weight").tensor()->shape()!=rt::Shape{7168,384})throw std::runtime_error("r120f MLA actual3head sharding");
    infinilm::cache::PagedKVCacheConfig cache(68,256,16);auto allocated=km::kimi_k3_allocate_cache_tensors(r,info,&cache,cfg,infinilm::backends::AttentionBackend::PAGED_ATTN);size_t cachebytes=0,kdas=0,mlas=0;J cache_shapes=J::array();
    for(size_t l=0;l<93;++l){auto kv=allocated.kv_cache_tensors.at(l),conv=allocated.conv_state_tensors.at(l),ssm=allocated.ssm_state_tensors.at(l);if(kv){++mlas;if(kv->shape()!=rt::Shape{2,68,256,3,192}||conv||ssm)throw std::runtime_error("r120f MLA cache actual3heads");cachebytes+=kv->nbytes();cache_shapes.push_back({{"layer",l},{"KV",kv->shape()},{"bytes",kv->nbytes()}});}else{++kdas;if(!conv||!ssm||conv->shape()!=rt::Shape{17,1152,3}||ssm->shape()!=rt::Shape{17,3,128,128})throw std::runtime_error("r120f H3 actualstate pools");cachebytes+=conv->nbytes()+ssm->nbytes();cache_shapes.push_back({{"layer",l},{"conv",conv->shape()},{"recurrent",ssm->shape()}});}}
    if(cachebytes!=kimi_tp32_r120f::CacheBytes||kdas!=69||mlas!=24)throw std::runtime_error("r120f all93 cache category count/bytes");row["cache"]={{"bytes",cachebytes},{"KDA_layers",kdas},{"MLA_layers",mlas},{"shapes",cache_shapes}};row["after_components_and_cache"]=memory();transport(r,info);row["real32_BF16_headwidth_allgather"]=true;row["after_collective"]=memory();r.syncStream();
   }r.syncStream();r.trimMemory();row["after_components_closed"]=memory();owner.reset();row["normal_Runtime_close"]=true;owners[local]=row;
  }catch(const std::exception&e){std::fprintf(stderr,"r120f ctor fatal n%d local%d %s\n",node,local,e.what());std::terminate();}});
  for(auto&t:threads)t.join();out["all8normal_before_communicators"]=true;
 }out.update({{"completed",true},{"owners",owners},{"scope","no-weight KDA/MLA/head plus all93 caches and metadata allgather only"},{"full93_model_parameters_constructed",false},{"forward_calls",0}});result=out.dump();return 0;
}catch(const std::exception&e){out["error"]=e.what();result=out.dump();return 1;}}
