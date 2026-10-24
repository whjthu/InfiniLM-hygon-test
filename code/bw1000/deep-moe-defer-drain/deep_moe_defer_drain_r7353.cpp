#include "deep_tail_SUM_reduce_r7211.hpp"
#include <cstdlib>
#include <cstring>
namespace deep_tail_sum_r7211 {
inline std::atomic<int> policy{0};inline std::once_flag initialized;
struct Counts{std::atomic<uint64_t> selected{0},bytes{0},elements{0},native_target{0},bypass{0};};inline std::array<Counts,8> counts;
inline void initialize(){std::call_once(initialized,[]{const char*p=std::getenv("INFINILM_DEEP_MOE_TAIL512_R7211");if(!p)return;if(std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("7211 env must exactly 0 or1");policy.store(p[0]-'0');});}
}
extern "C" int deep_moe_tail_get_r7211(){try{deep_tail_sum_r7211::initialize();return deep_tail_sum_r7211::policy.load();}catch(...){return -1;}}
extern "C" int deep_moe_tail_set_r7211(int p){if(p!=0&&p!=1)return 1;try{deep_tail_sum_r7211::initialize();namespace m=infinilm::models::deepseek_v4;std::lock_guard<std::mutex>lock(m::wide_r3443::policy_mutex);if(m::wide_r3443::in_flight)return 2;deep_tail_sum_r7211::policy.store(p);return 0;}catch(...){return 3;}}
extern "C" uint64_t deep_moe_tail_calls_r7211(int d,int k){if(d<0||d>=8||k<0||k>4)return UINT64_MAX;auto&c=deep_tail_sum_r7211::counts[d];return k==0?c.selected.load():k==1?c.bytes.load():k==2?c.elements.load():k==3?c.native_target.load():c.bypass.load();}
#include "deep_collective_layout_r7040.hpp"
#include "deep_moe_defer_drain_r7353.hpp"
#include "deep_reduce_wide_r3443.hpp"
#include "deepseek_v4_moe_routed_r20.hpp"
#include "deepseek_routed_mlp_batch_r20.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
extern "C" int deep_prefill_scatter_r194(void*,const void*,const void*,const void*,size_t,size_t,void*);
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {void trace(const char *stage){if(std::getenv("DEEPSEEK_MOE_TRACE")){std::fprintf(stderr,"DEEPSEEK_MOE_NATIVE_STAGE %s\n",stage);std::fflush(stderr);}}}
DeepseekV4MoERoutedR20::DeepseekV4MoERoutedR20(rt::Runtime &r,size_t hidden,size_t intermediate,size_t shared_intermediate,size_t experts,size_t topk,size_t vocab,bool hash,double factor,const engine::distributed::RankInfo &rank)
 :device_(r.device()),hidden_(hidden),experts_(experts),topk_(topk),vocab_(vocab),hash_(hash),tp_size_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||hidden!=4096||experts!=256||topk!=6||!vocab||!std::isfinite(factor)||factor<=0||(tp_size_!=1&&tp_size_!=8)||(tp_size_>1&&!comm_))throw std::invalid_argument("DeepSeek eager MoE requires explicit Hygon TP1/TP8 ownership and real model geometry");
 try{
  gate_=rt::Tensor::empty_parameter(r,{experts,hidden},rt::DataType::kBFloat16);bind_weight("gate.weight",weights::Weight(gate_));
  if(hash){table_=rt::Tensor::empty_parameter(r,{vocab,topk},rt::DataType::kInt64);bind_weight("gate.tid2eid",weights::Weight(table_));}
  else{bias_=rt::Tensor::empty_parameter(r,{experts},rt::DataType::kFloat32);bind_weight("gate.bias",weights::Weight(bias_));}
  if(tp_size_!=8||intermediate!=2048)throw std::invalid_argument("DeepSeek routed bank candidate requires actual TP8 experts");
  w13_bank_=rt::Tensor::empty_parameter(r,{256,512,4096},rt::DataType::kInt8);
  s13_bank_=rt::Tensor::empty_parameter(r,{256,512,1},rt::DataType::kFloat32);
  w2_bank_=rt::Tensor::empty_parameter(r,{256,4096,256},rt::DataType::kInt8);
  s2_bank_=rt::Tensor::empty_parameter(r,{256,4096,1},rt::DataType::kFloat32);
  for(size_t i=0;i<experts;++i){
   auto slice=[&](const rt::Tensor&t,rt::Shape shape){return t->narrow({{0,i,1}})->view(shape);};
   auto child=std::make_shared<DeepseekV4Int8MLPBankR20>(r,hidden,intermediate,rt::DataType::kBFloat16,rank,
      slice(w13_bank_,{512,4096}),slice(s13_bank_,{512,1}),slice(w2_bank_,{4096,256}),slice(s2_bank_,{4096,1}));
   experts_modules_.push_back(bind_child("experts."+std::to_string(i),child));
  }
  shared_=bind_child("shared_experts",std::make_shared<DeepseekV4Int8MLPBankR20>(r,hidden,shared_intermediate,rt::DataType::kBFloat16,rank));
  float eps=1e-20f,scale=static_cast<float>(factor);
  epsilon_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);factor_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);
  auto he=rt::Tensor::from_blob(&eps,{1},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}),hf=rt::Tensor::from_blob(&scale,{1},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0});
  epsilon_->copy_from(r,he);factor_->copy_from(r,hf);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void DeepseekV4MoERoutedR20::process_weights_after_loading(rt::Runtime &r,bool){r.assertOwnerThread();if(ready_||r.device()!=device_||r.isGraphRecording())throw std::logic_error("DeepSeek eager MoE postload state differs");r.syncStream();ready_=true;}
DeepseekV4MoERoutedR20::Output DeepseekV4MoERoutedR20::forward(rt::Runtime &r,const rt::Tensor &x,const rt::Tensor &token_ids)const{
 r.assertOwnerThread();if(!ready_||r.isGraphRecording()||r.device()!=device_||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=hidden_)throw std::invalid_argument("DeepSeek eager MoE requires loaded packed BF16 input");
 size_t tokens=x->size(0);auto alloc=[&](rt::Shape shape,rt::DataType type){return rt::Tensor::empty(r,shape,type);};
 auto logits=alloc({tokens,experts_},rt::DataType::kFloat32),soft=alloc({tokens,experts_},rt::DataType::kFloat32),scores=alloc({tokens,experts_},rt::DataType::kFloat32);
 trace("gate_projection");ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(device_.type()),x,gate_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits);
 ex::submit<infini::ops::Softplus>(r,infini::ops::Softplus::DefaultConfig(device_.type()),logits,double{1},double{20},soft);
 ex::submit<infini::ops::Sqrt>(r,infini::ops::Sqrt::DefaultConfig(device_.type()),soft,scores);
 auto indices=alloc({tokens,topk_},rt::DataType::kInt64),selected_weights=alloc({tokens,topk_},rt::DataType::kFloat32);
 trace("select_routes");if(hash_){
  if(!token_ids||token_ids->device()!=device_||token_ids->dtype()!=rt::DataType::kInt64||token_ids->shape()!=rt::Shape{tokens})throw std::invalid_argument("DeepSeek hash routing requires packed token IDs");
  std::vector<int64_t> ids(tokens);r.syncStream();r.memcpyD2H(ids.data(),token_ids->data(),tokens*sizeof(int64_t));for(auto id:ids)if(id<0||size_t(id)>=vocab_)throw std::invalid_argument("DeepSeek token ID outside hash table");
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(device_.type()),table_,token_ids,int64_t{0},indices);
 }else{
  auto selection=alloc({tokens,experts_},rt::DataType::kFloat32),values=alloc({tokens,topk_},rt::DataType::kFloat32);
  ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),scores,bias_,selection);
  ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(device_.type()),selection,int64_t(topk_),int64_t{1},true,true,values,indices);
 }
 ex::submit<infini::ops::Gather>(r,infini::ops::Gather::DefaultConfig(device_.type()),scores,indices,int64_t{1},false,selected_weights);
 auto sum=alloc({tokens,1},rt::DataType::kFloat32),denominator=alloc({tokens,1},rt::DataType::kFloat32),normalized=alloc({tokens,topk_},rt::DataType::kFloat32),route_weights=alloc({tokens,topk_},rt::DataType::kFloat32);
 ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(device_.type()),selected_weights,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},sum);
 ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),sum,epsilon_,denominator);
 ex::submit<infini::ops::Div>(r,infini::ops::Div::DefaultConfig(device_.type()),selected_weights,denominator,normalized);
 ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(device_.type()),normalized,factor_,route_weights);
 rt::Tensor routed;
 if(tokens<=64){
  trace("gpu_batched_int8_routes");
  routed=routed_r20::forward(r,x,indices,route_weights,w13_bank_,s13_bank_,w2_bank_,s2_bank_);
 }else{
 trace("read_route_metadata");std::vector<int64_t> routes(tokens*topk_);r.syncStream();r.memcpyD2H(routes.data(),indices->data(),routes.size()*sizeof(int64_t));
 std::map<size_t,std::vector<size_t>> grouped;for(size_t i=0;i<routes.size();++i){if(routes[i]<0||size_t(routes[i])>=experts_)throw std::invalid_argument("DeepSeek selected expert outside pool");grouped[size_t(routes[i])].push_back(i);}
 auto contributions=alloc({tokens*topk_,hidden_},rt::DataType::kFloat32);
 for(const auto &[expert,positions]:grouped){
  size_t rows=positions.size();std::vector<int64_t> row_ids(rows),route_ids(rows);for(size_t j=0;j<rows;++j){row_ids[j]=positions[j]/topk_;route_ids[j]=positions[j];}
  auto upload=[&](std::vector<int64_t>&v){auto host=rt::Tensor::from_blob(v.data(),{v.size()},rt::DataType::kInt64,{rt::Device::Type::kCpu,0});auto out=alloc({v.size()},rt::DataType::kInt64);out->copy_from(r,host);r.syncStream();return out;};
  trace("expert_input_gather");auto row_index=upload(row_ids),route_index=upload(route_ids);auto gathered=alloc({rows,hidden_},rt::DataType::kBFloat16);
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(device_.type()),x,row_index,int64_t{0},gathered);
  trace("expert_int8_mlp");auto local=experts_modules_.at(expert)->forward_local(r,gathered);
  int status=deep_prefill_scatter_r194(contributions->data(),local->data(),route_index->data(),route_weights->data(),rows,tokens*topk_,r.stream());
  if(status)throw std::runtime_error(std::string("DeepSeek prefill scatter: ")+cudaGetErrorString(static_cast<cudaError_t>(status)));
  r.syncStream();
 }
 trace("sum_routed");routed=alloc({tokens,hidden_},rt::DataType::kFloat32);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(device_.type()),contributions->view({tokens,topk_,hidden_}),std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},routed);
  // Keep every fallback contribution alive until its Sum has completed.
  r.syncStream();
 }
 trace("shared_int8_mlp");auto shared=shared_->forward_local(r,x),shared_float=alloc({tokens,hidden_},rt::DataType::kFloat32),combined=alloc({tokens,hidden_},rt::DataType::kFloat32);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),shared,shared_float);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),routed,shared_float,combined);
 trace("tp_reduction");if(tp_size_>1){
  deep_tail_sum_r7211::initialize();const bool target=deep_tail_prefill_r7211::target(r,combined);const bool selected=target&&deep_tail_sum_r7211::policy.load()==1;auto&counter=deep_tail_sum_r7211::counts.at(r.device().index());
  if(selected){auto before=layout_r7040::counts.at(r.device().index()).pad_bytes.load();layout_r7040::reduce_moe_tail_r7211(r,combined,comm_,true);auto bytes=layout_r7040::counts.at(r.device().index()).pad_bytes.load()-before;if(bytes!=2048*4096*4)throw std::logic_error("7211 actual canonical payload differs");++counter.selected;counter.bytes.fetch_add(bytes);counter.elements.fetch_add(bytes/4);}
  else{defer_drain_r7353::reduce_moe(r,combined,comm_);if(target)++counter.native_target;else ++counter.bypass;}
 }
 auto output=alloc({tokens,hidden_},rt::DataType::kBFloat16);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),combined,output);return {output,route_weights,indices};
}
}
