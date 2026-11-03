"""Exact current9872 source transformation; no expert/TP/hash math substitution."""
SOURCE_SHA='9872e3553ee22c1292637559a116fe8c0fd9726aaa1290206a3bfc6beb5117c6'
HEADER_SHA='b14a666b2c7d28360a678f9ac348fb10c3e3b436de833939e26b719951ea77e6'
def rebase(source,header):
 source=source.split('\nextern "C" int deep_model_activation_arm_r8846',1)[0]
 source='\n'.join(line for line in source.splitlines()if not line.startswith(('extern "C" int deep_moe_tail_get_r7211','extern "C" int deep_moe_tail_set_r7211','extern "C" uint64_t deep_moe_tail_calls_r7211')))+'\n'
 source=source.replace('namespace deep_tail_sum_r7211','namespace deep_tail_sum_r419').replace('deep_tail_sum_r7211::','deep_tail_sum_r419::')
 source=source.replace('namespace infinilm::models::deepseek_v4 {','namespace infinilm::models::deepseek_v4::router_suffix_r419 {')
 source=source.replace('namespace rt=infinilm::runtime;namespace ex=infinilm::execution;','// aliases are provided by the shared router_suffix.hpp')
 source=source.replace('#include "deepseek_v4_moe_routed_r20.hpp"','#include "private_moe.hpp"\n#include "router_suffix.hpp"')
 old='auto logits=alloc({tokens,experts_},rt::DataType::kFloat32),soft=alloc({tokens,experts_},rt::DataType::kFloat32),scores=alloc({tokens,experts_},rt::DataType::kFloat32);'
 assert source.count(old)==1
 new='auto logits=alloc({tokens,experts_},rt::DataType::kFloat32);deep419::Geometry geometry{tokens,experts_,topk_,hash_,true,bias_&&bias_->dtype()==rt::DataType::kFloat32,epsilon_->dtype()==rt::DataType::kFloat32&&factor_->dtype()==rt::DataType::kFloat32,true,logits->is_contiguous()&&(!bias_||bias_->is_contiguous())};bool fused=deep419::select(geometry);auto router=router_buffers(r,tokens,experts_,fused);'
 source=source.replace(old,new)
 start=source.index(' ex::submit<infini::ops::Softplus>',source.index('DeepseekV4MoERoutedR20::Output DeepseekV4MoERoutedR20::forward'))
 end=source.index(' route_span.close();',start)
 original_suffix=source[start:end]
 assert original_suffix.count('ex::submit<')==10 # includes the mutually exclusive hash IndexSelect vs learnedAdd/TopK
 replacement=' router_suffix(r,logits,bias_,table_,token_ids,epsilon_,factor_,vocab_,hash_,topk_,fused,router);auto&indices=router.indices;auto&route_weights=router.weights;\n'
 source=source[:start]+replacement+source[end:]
 gate='ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(device_.type()),x,gate_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits);'
 assert source.count(gate)==1;source=source.replace(gate,'gate_project(r,x,gate_,logits);')
 marker=' Output forward(runtime::Runtime &,const runtime::Tensor &,const runtime::Tensor &) const;'
 assert header.count(marker)==1
 header=header.replace(marker,marker+'\n const runtime::Tensor& diagnostic_gate()const{return gate_;}\n const runtime::Tensor& diagnostic_bias()const{return bias_;}\n const runtime::Tensor& diagnostic_epsilon()const{return epsilon_;}\n const runtime::Tensor& diagnostic_factor()const{return factor_;}')
 header=header.replace('namespace infinilm::models::deepseek_v4 {','namespace infinilm::models::deepseek_v4::router_suffix_r419 {')
 return source,header,original_suffix
