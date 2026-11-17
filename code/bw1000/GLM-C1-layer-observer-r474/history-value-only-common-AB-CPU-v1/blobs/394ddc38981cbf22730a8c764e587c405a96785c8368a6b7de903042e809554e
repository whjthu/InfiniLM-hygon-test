#include "geometry.hpp"
#include "attention_detail_v1.hpp"
#include "glm_layer_trace_r5670.hpp"
#include "glm_decode_leases_r1354.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "glm_moe_dsa_attention_output.hpp"
#include "execution/submit.hpp"
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;int entered_exceptions=std::uncaught_exceptions();~Work(){decode_leases_r1354::retire(r,held,entered_exceptions);}rt::Tensor alloc(rt::Shape shape,rt::DataType type){auto t=rt::Tensor::empty(r,shape,type);held.push_back(t);return t;}};}
AttentionOutput::AttentionOutput(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank):owner_(&r),columns_(0),offset_(0),tp_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(tp_!=1&&tp_!=8&&!glm_global_tp16_r470::admitted(rank))||rank.tp_rank<0||rank.tp_rank>=tp_||(tp_>1&&!comm_)||cfg.at("hidden_size")!=6144||cfg.at("num_attention_heads")!=64||cfg.at("v_head_dim")!=256||cfg.value("attention_bias",false))throw std::invalid_argument("GLM attention output geometry/owner/TP differs");
 columns_=16384/size_t(tp_);offset_=size_t(rank.tp_rank)*columns_;
 try{weight_=rt::Tensor::empty_parameter(r,{6144,columns_},rt::DataType::kUInt8);scale_=rt::Tensor::empty_parameter(r,{48,128},rt::DataType::kFloat32);bind_weight("o_proj.weight",weights::Weight(weight_,1,rank.tp_rank,tp_));bind_weight("o_proj.weight_scale_inv",weights::Weight(scale_,-1,rank.tp_rank,tp_));}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void AttentionOutput::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(owner_!=&r||ready_||r.isGraphRecording())throw std::logic_error("GLM output postload owner/state differs");r.syncStream();ready_=true;}
AttentionOutput::Output AttentionOutput::forward(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();if(owner_!=&r||!ready_||r.isGraphRecording()||!x||x->device()!=r.device()||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=3||!x->size(0)||x->size(1)!=local_heads()||x->size(2)!=256||!x->is_contiguous())throw std::invalid_argument("GLM attention output requires contiguous local BF16 V256 heads");
 Work work{r};work.held.push_back(x);size_t tokens=x->size(0);auto input=x->view({tokens,columns_});auto packed=work.alloc(input->shape(),rt::DataType::kUInt8),local_scale=work.alloc({tokens,columns_/128},rt::DataType::kFloat32),global_scale=work.alloc({tokens,128},rt::DataType::kFloat32),local=work.alloc({tokens,6144},rt::DataType::kBFloat16);

/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::tuple(r,"output",tokens,size_t{6144},columns_,size_t{0},offset_);
/* ATTND1_OBSERVATION_END */
 // Each TP shard starts and ends on a 128-element activation block. Local
 // quantization is therefore identical to quantizing before the split. Keep
 // scale indexing in the original global blocks used by the checkpoint.
glm_layer_mark_r5670(r,"attnout.quant.begin");
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.quant.begin");
/* ATTND1_OBSERVATION_END */
ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),input,packed,local_scale);
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.quant.return");
/* ATTND1_OBSERVATION_END */


/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.scale_scatter.begin");
/* ATTND1_OBSERVATION_END */
r.setDeviceMemory(global_scale->data(),0xff,global_scale->nbytes());global_scale->narrow({{1,offset_/128,columns_/128}})->copy_from(r,local_scale);
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.scale_scatter.return");
/* ATTND1_OBSERVATION_END */

glm_layer_mark_r5670(r,"attnout.projection.begin");
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.projection.begin");
/* ATTND1_OBSERVATION_END */
const bool ad_v1_cached = fp8_cached_r809(r,local,packed,global_scale,weight_,scale_,size_t{0},offset_);
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::dispatch(r,"output",ad_v1_cached);
/* ATTND1_OBSERVATION_END */
 if(!ad_v1_cached)ex::submit<infini::ops::Fp8BlockLinear>(r,ex::RecordingPolicy::HostOnly,infini::ops::Fp8BlockLinear::DefaultConfig(r.device().type()),local,packed,global_scale,weight_,scale_,size_t{0},offset_);

/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.projection.return");
/* ATTND1_OBSERVATION_END */
glm_layer_mark_r5670(r,"attnout.TP.begin"); auto hidden=local;if(tp_>1){hidden=work.alloc(local->shape(),local->dtype());
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.copy.begin");
/* ATTND1_OBSERVATION_END */
hidden->copy_from(r,local);
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.copy.return");
/* ATTND1_OBSERVATION_END */

/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.allreduce.begin");
/* ATTND1_OBSERVATION_END */
engine::distributed::allreduce_(r,hidden,hidden,infinicclSum,comm_);
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.allreduce.return");
/* ATTND1_OBSERVATION_END */
}glm_layer_mark_r5670(r,"attnout.TP.end");
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.original_success_fence.begin");
/* ATTND1_OBSERVATION_END */
decode_leases_r1354::success_fence(r);
/* ATTND1_OBSERVATION_BEGIN */
glm_attention_detail_v1::mark(r,"attnd1.output.original_success_fence.return");
/* ATTND1_OBSERVATION_END */
return {local,hidden};
}
}
