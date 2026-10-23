#include "deep_collective_layout_r7040.hpp"
#include <atomic>
#include <array>
#include <cstdlib>
#include <cstring>
#include "deep_reduce_wide_r3443.hpp"
#include "deepseek_v4_attention_output.hpp"
#include "execution/submit.hpp"
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8>tile_calls_r4808{};
std::array<std::array<std::atomic<uint64_t>,3>,8>direct_stats_r7247{};}
namespace {struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}};}
AttentionOutput::AttentionOutput(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank):device_(r.device()),groups_(0),tp_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||(tp_!=1&&tp_!=8)||rank.tp_rank<0||rank.tp_rank>=tp_||(tp_>1&&!comm_)||cfg.at("hidden_size")!=4096||cfg.at("head_dim")!=512||cfg.at("num_attention_heads")!=64||cfg.at("o_groups")!=8||cfg.at("o_lora_rank")!=1024)throw std::invalid_argument("DeepSeek output checkpoint geometry/owner/TP differs");
 groups_=8/size_t(tp_);
 try{
 a_=rt::Tensor::empty_parameter(r,{groups_*1024,4096},rt::DataType::kBFloat16);
 b_=rt::Tensor::empty_parameter(r,{4096,groups_*1024},rt::DataType::kInt8);
 bs_=rt::Tensor::empty_parameter(r,{4096,1},rt::DataType::kFloat32);
 bind_weight("wo_a.weight",weights::Weight(a_,0,rank.tp_rank,tp_));
 bind_weight("wo_b.weight",weights::Weight(b_,1,rank.tp_rank,tp_));
 bind_weight("wo_b.scale",weights::Weight(bs_,-1,rank.tp_rank,tp_));
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void AttentionOutput::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(ready_||r.device()!=device_||r.isGraphRecording())throw std::logic_error("DeepSeek output postload state differs");r.syncStream();ready_=true;}
AttentionOutput::Output AttentionOutput::forward(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();
 if(!ready_||r.device()!=device_||r.isGraphRecording()||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||!x->is_contiguous()||x->ndim()!=3||!x->size(0)||x->size(1)!=groups_*8||x->size(2)!=512)throw std::invalid_argument("DeepSeek output requires local contiguous inverse-rotated attention heads");
 Work work{r};size_t tokens=x->size(0);
 auto grouped=work.alloc({groups_,tokens,1024},rt::DataType::kBFloat16);

 const char*tile_env=std::getenv("INFINILM_DEEP_OUTPUT_TILE_R4808");int tile=tile_env?std::atoi(tile_env):0;
 if(tile!=0&&tile!=16)throw std::invalid_argument("4808 invalid projection tile");
 // Configure only at a driver quiescence barrier, like existing tile_env.
 const char*direct_env=std::getenv("INFINILM_DEEP_OUTPUT_DIRECT_R7247");
 if(direct_env&&std::strcmp(direct_env,"0")&&std::strcmp(direct_env,"1"))throw std::invalid_argument("7247 invalid direct projection flag");
 const bool direct=direct_env&&std::strcmp(direct_env,"1")==0&&tp_==8&&groups_==1&&tile==16&&tokens>16&&tokens<=64;
 if(tile==16&&tokens>16&&tokens<=64){
  for(size_t begin=0;begin<tokens;begin+=16){size_t width=std::min(size_t{16},tokens-begin);rt::Tensor part;
   if(direct){
    // Singleton strides are significant in this Runtime. Drop group dim before
    // restoring it, yielding exactly the old part's canonical shape/strides.
    part=grouped->narrow({{1,begin,width}})->squeeze(0)->view({1,width,1024});
    if(!part->is_contiguous()||part->strides()!=rt::Strides{rt::Stride(width*1024),1024,1})throw std::logic_error("7247 noncanonical destination");
    work.held.push_back(part); // parent grouped and view survive owner drain
   }else part=work.alloc({groups_,width,1024},rt::DataType::kBFloat16);
   auto input=x->narrow({{0,begin,width}})->contiguous(r);
   ex::submit<infini::ops::Gemm>(r,ex::RecordingPolicy::DeviceOnly,infini::ops::Gemm::DefaultConfig(device_.type()),input->view({width,groups_,4096})->permute({1,0,2}),a_->view({groups_,1024,4096})->permute({0,2,1}),std::optional<rt::Tensor>{},std::optional<float>{1.f},std::optional<float>{0.f},std::optional<int>{},std::optional<int>{},part);
   if(!direct)grouped->narrow({{1,begin,width}})->copy_from(r,part);
  }tile_calls_r4808.at(r.device().index()).fetch_add(1);
  if(direct){auto&c=direct_stats_r7247.at(r.device().index());++c[0];c[1].fetch_add((tokens+15)/16);c[2].fetch_add(tokens*1024*2);}
 }else{
 ex::submit<infini::ops::Gemm>(r,ex::RecordingPolicy::DeviceOnly,infini::ops::Gemm::DefaultConfig(device_.type()),x->view({tokens,groups_,4096})->permute({1,0,2}),a_->view({groups_,1024,4096})->permute({0,2,1}),std::optional<rt::Tensor>{},std::optional<float>{1.f},std::optional<float>{0.f},std::optional<int>{},std::optional<int>{},grouped);
 }
 auto flat=grouped->permute({1,0,2})->contiguous(r)->view({tokens,groups_*1024});work.held.push_back(flat);
 auto packed=work.alloc(flat->shape(),rt::DataType::kInt8),scale=work.alloc({tokens,1},rt::DataType::kFloat32),local=work.alloc({tokens,4096},rt::DataType::kBFloat16);
 auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(device_.type());quant.set_implementation_index(0);
 auto gemm=infini::ops::I8Gemm::DefaultConfig(device_.type());gemm.set_implementation_index(0);
 ex::submit<infini::ops::PerChannelQuantI8>(r,ex::RecordingPolicy::HostOnly,quant,flat,packed,scale);
 ex::submit<infini::ops::I8Gemm>(r,ex::RecordingPolicy::HostOnly,gemm,local,packed,scale,b_->permute({1,0}),bs_,std::optional<rt::Tensor>{});
 auto hidden=local;
 if(tp_>1){hidden=work.alloc(local->shape(),local->dtype());hidden->copy_from(r,local);layout_r7040::reduce_attention(r,hidden,comm_);}
 r.syncStream();return {flat,local,hidden};
}
}

extern "C" int modern_deep_precision_policy_r3443(int flag){if(flag!=0&&flag!=1&&flag!=5)return 1;return infinilm::models::deepseek_v4::wide_r3443::set(flag);}
extern "C" int modern_deep_precision_stats_r3443(int d,uint64_t*out,size_t n){return infinilm::models::deepseek_v4::wide_r3443::stats(d,out,n);}

extern "C" uint64_t modern_deep_FP32_calls_r3597(int device){if(device<0||device>=8)return ~uint64_t{};return infinilm::models::deepseek_v4::wide_r3443::fp32_calls.at(device).load();}

extern "C" uint64_t deep_output_tile_calls_r4808(int d){return d>=0&&d<8?infinilm::models::deepseek_v4::tile_calls_r4808[size_t(d)].load():~uint64_t{};}

extern "C" uint64_t deep_output_direct_stats_r7247(int d,int k){return d>=0&&d<8&&k>=0&&k<3?infinilm::models::deepseek_v4::direct_stats_r7247[size_t(d)][size_t(k)].load():~uint64_t{};}
