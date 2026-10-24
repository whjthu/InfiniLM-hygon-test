#include "deep_collective_layout_r7040.hpp"
#include <atomic>
#include <array>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include "deep_reduce_wide_r3443.hpp"
#include "deepseek_v4_attention_output.hpp"
#include "execution/submit.hpp"
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8>tile_calls_r4808{};}
namespace output_clock_r8713 {
struct Slot{int device=-1;cudaEvent_t before{},after{};bool enabled=false,valid=false;uint64_t sequence=0;double cpu[5]{};double event_seconds=0;size_t tokens=0;};
thread_local Slot slot;thread_local std::string error;
inline double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline void gpu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline bool selected(rt::Runtime&r){if(!slot.enabled)return false;if(slot.device!=r.device().index())throw std::logic_error("8713 foreign device");auto*f=layout_r7040::core::current();return wide_r3443::selected(r)==5&&(!f||!(f->policy&2)||f->requests!=1);}
inline void reduce(rt::Runtime&r,const rt::Tensor&v,infinicclComm_t comm){
 layout_r7040::validate(r,v,comm);rt::Tensor scratch;
 try{
  scratch=rt::Tensor::empty(r,v->shape(),rt::DataType::kFloat32);
  auto cfg=infini::ops::Cast::DefaultConfig(r.device().type());
  ex::submit<infini::ops::Cast>(r,cfg,v,scratch);
  gpu(cudaEventRecord(slot.before,static_cast<cudaStream_t>(r.stream())));
  slot.cpu[2]=now();
  infinilm::engine::distributed::allreduce_(r,scratch,scratch,infinicclSum,comm);
  slot.cpu[3]=now();gpu(cudaEventRecord(slot.after,static_cast<cudaStream_t>(r.stream())));
  ex::submit<infini::ops::Cast>(r,cfg,scratch,v);r.syncStream();
  slot.cpu[4]=now();float ms=0;gpu(cudaEventElapsedTime(&ms,slot.before,slot.after));slot.event_seconds=ms*.001;slot.valid=true;++slot.sequence;
  auto&z=wide_r3443::counts.at(r.device().index()).v;++z[0];++z[1];++wide_r3443::fp32_calls.at(r.device().index());const auto bytes=scratch->nbytes();z[4].fetch_add(bytes);auto peak=z[5].load();while(peak<bytes&&!z[5].compare_exchange_weak(peak,bytes)){};
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
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
 const bool observe=tp_==8&&groups_==1&&output_clock_r8713::selected(r);if(observe){auto&z=output_clock_r8713::slot;z.valid=false;z.cpu[0]=output_clock_r8713::now();z.tokens=tokens;}
 auto grouped=work.alloc({groups_,tokens,1024},rt::DataType::kBFloat16);

 const char*tile_env=std::getenv("INFINILM_DEEP_OUTPUT_TILE_R4808");int tile=tile_env?std::atoi(tile_env):0;
 if(tile!=0&&tile!=16)throw std::invalid_argument("4808 invalid projection tile");
 if(tile==16&&tokens>16&&tokens<=64){
  for(size_t begin=0;begin<tokens;begin+=16){size_t width=std::min(size_t{16},tokens-begin);auto part=work.alloc({groups_,width,1024},rt::DataType::kBFloat16);auto input=x->narrow({{0,begin,width}})->contiguous(r);
   ex::submit<infini::ops::Gemm>(r,ex::RecordingPolicy::DeviceOnly,infini::ops::Gemm::DefaultConfig(device_.type()),input->view({width,groups_,4096})->permute({1,0,2}),a_->view({groups_,1024,4096})->permute({0,2,1}),std::optional<rt::Tensor>{},std::optional<float>{1.f},std::optional<float>{0.f},std::optional<int>{},std::optional<int>{},part);
   grouped->narrow({{1,begin,width}})->copy_from(r,part);
  }tile_calls_r4808.at(r.device().index()).fetch_add(1);
 }else{
 ex::submit<infini::ops::Gemm>(r,ex::RecordingPolicy::DeviceOnly,infini::ops::Gemm::DefaultConfig(device_.type()),x->view({tokens,groups_,4096})->permute({1,0,2}),a_->view({groups_,1024,4096})->permute({0,2,1}),std::optional<rt::Tensor>{},std::optional<float>{1.f},std::optional<float>{0.f},std::optional<int>{},std::optional<int>{},grouped);
 }
 auto flat=grouped->permute({1,0,2})->contiguous(r)->view({tokens,groups_*1024});work.held.push_back(flat);
 auto packed=work.alloc(flat->shape(),rt::DataType::kInt8),scale=work.alloc({tokens,1},rt::DataType::kFloat32),local=work.alloc({tokens,4096},rt::DataType::kBFloat16);
 auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(device_.type());quant.set_implementation_index(0);
 auto gemm=infini::ops::I8Gemm::DefaultConfig(device_.type());gemm.set_implementation_index(0);
 ex::submit<infini::ops::PerChannelQuantI8>(r,ex::RecordingPolicy::HostOnly,quant,flat,packed,scale);
 ex::submit<infini::ops::I8Gemm>(r,ex::RecordingPolicy::HostOnly,gemm,local,packed,scale,b_->permute({1,0}),bs_,std::optional<rt::Tensor>{});
 if(observe)output_clock_r8713::slot.cpu[1]=output_clock_r8713::now();
 auto hidden=local;
 if(tp_>1){hidden=work.alloc(local->shape(),local->dtype());hidden->copy_from(r,local);if(observe)output_clock_r8713::reduce(r,hidden,comm_);else layout_r7040::reduce_attention(r,hidden,comm_);}
 r.syncStream();return {flat,local,hidden};
}
}

extern "C" int modern_deep_precision_policy_r3443(int flag){if(flag!=0&&flag!=1&&flag!=5)return 1;return infinilm::models::deepseek_v4::wide_r3443::set(flag);}
extern "C" int modern_deep_precision_stats_r3443(int d,uint64_t*out,size_t n){return infinilm::models::deepseek_v4::wide_r3443::stats(d,out,n);}

extern "C" uint64_t modern_deep_FP32_calls_r3597(int device){if(device<0||device>=8)return ~uint64_t{};return infinilm::models::deepseek_v4::wide_r3443::fp32_calls.at(device).load();}

extern "C" uint64_t deep_output_tile_calls_r4808(int d){return d>=0&&d<8?infinilm::models::deepseek_v4::tile_calls_r4808[size_t(d)].load():~uint64_t{};}

extern "C" int deep_output_clock_config_r8713(int device,int mode){using namespace infinilm::models::deepseek_v4::output_clock_r8713;auto&z=slot;try{int current=-1;gpu(cudaGetDevice(&current));if(device!=current||mode<0||mode>2||(z.device>=0&&z.device!=device))throw std::invalid_argument("8713 current device/mode differs");if(mode==2){z.enabled=false;z.valid=false;if(z.before){gpu(cudaEventDestroy(z.before));z.before=nullptr;}if(z.after){gpu(cudaEventDestroy(z.after));z.after=nullptr;}z.device=-1;return 0;}if(!z.before){gpu(cudaEventCreate(&z.before));try{gpu(cudaEventCreate(&z.after));}catch(...){gpu(cudaEventDestroy(z.before));z.before=nullptr;throw;}z.device=device;}z.enabled=mode==1;z.valid=false;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int deep_output_clock_read_r8713(double*out,size_t n){using namespace infinilm::models::deepseek_v4::output_clock_r8713;auto&z=slot;if(!out||n!=8||!z.valid)return 1;for(size_t i=0;i<5;++i)out[i]=z.cpu[i];out[5]=z.event_seconds;out[6]=double(z.sequence);out[7]=double(z.tokens);return 0;}
extern "C" const char*deep_output_clock_error_r8713(){return infinilm::models::deepseek_v4::output_clock_r8713::error.c_str();}
