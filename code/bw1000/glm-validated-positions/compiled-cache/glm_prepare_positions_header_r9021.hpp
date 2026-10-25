#pragma once
#include "/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/weights/weights.hpp"
#include "/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/runtime/runtime.hpp"
#include "/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/layers/normalization/rmsnorm.hpp"
#include "/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/layers/rotary_embedding/rope.hpp"
#include "/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/engine/distributed/communication_group.hpp"
#include <nlohmann/json.hpp>
namespace infinilm::models::glm_moe_dsa {
class Fp8Projection final:public weights::ModelWeights{
public:
 Fp8Projection(runtime::Runtime&,size_t,size_t,int,int);
 runtime::Tensor forward(runtime::Runtime&,const runtime::Tensor&)const;
private:
 runtime::Device device_;size_t rows_,columns_,offset_;runtime::Tensor weight_,scale_;
};
class AttentionPrepare final:public weights::ModelWeights{
public:
 struct Output{runtime::Tensor query,key,value,q_resid,kv_latent,k_rot;};
 AttentionPrepare(runtime::Runtime&,const nlohmann::json&,const engine::distributed::RankInfo&);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&)const;
 Output forward_latent(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&)const;
 std::pair<runtime::Tensor,runtime::Tensor> expand_cached(runtime::Runtime&,const runtime::Tensor&latent,const runtime::Tensor&rotary_key)const;
 size_t local_heads()const{return heads_;}
private:
 friend class NativeAttentionProjectedR22;
 Output forward_latent_validated_r9021(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const std::vector<int64_t>&)const;
 Output prepare_impl_validated_r9021(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,bool,const std::vector<int64_t>*)const;
 Output prepare_impl(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,bool expand)const;
 runtime::Device device_;size_t heads_,maximum_;bool ready_=false;
 std::shared_ptr<Fp8Projection>qa_,qb_,ka_,kb_;
 runtime::Tensor qn_,kn_,qones_,kones_;std::shared_ptr<layers::RoPE>rope_;
};
}
