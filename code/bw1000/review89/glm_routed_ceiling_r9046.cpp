#include "glm_vector_dispatch_r8540.hpp"
#include "glm_layer_trace_r5670.hpp"
#include <fstream>
#include <atomic>
#include <array>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include "glm_decode_leases_r1354.hpp"
#include "glm_moe_dsa_routed_ffn_r19.hpp"
#include "execution/submit.hpp"
#include <map>
#include <cmath>
#include <stdexcept>
extern "C" int glm_grouped_gate_r5123(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_grouped_down_r5123(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {

// Fixed-fixture diagnostic only. Default OFF, no production reuse policy.
struct RoutedCeiling9046 { int mode=0; rt::Runtime*owner=nullptr; const void*module=nullptr; size_t offset=0; rt::Tensor local,current_x,current_ids; std::vector<unsigned char>reference_x,reference_ids; uint64_t counts[3]={0,0,0}; };
thread_local RoutedCeiling9046 routed_ceiling9046;
struct ActualCaptureState5164{std::atomic<bool>armed{false};std::mutex mutex;std::string prefix;};
std::array<ActualCaptureState5164,8> actual_capture_states5164;
int actual_capture_arm5164(int device,const char*prefix){
 if(device<0||device>=8||!prefix)return-1;
 std::string value(prefix);if(value.size()>260)return-2;
 if(!value.empty()){
  const char*enabled=std::getenv("GLM_ACTUAL_CAPTURE_R5164");
  if(!enabled||std::strcmp(enabled,"1")||value.rfind("/refactor/results/glm-actual-capture-r5164-",0)!=0||value.find("..")!=std::string::npos)return-3;
  for(char c:value)if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='/'||c=='-'||c=='_'||c=='.'))return-4;
 }
 auto&s=actual_capture_states5164[device];std::lock_guard<std::mutex>lock(s.mutex);s.prefix=value;s.armed.store(!value.empty(),std::memory_order_release);return 0;
}
struct ActualCaptureFrame5164{
 std::string prefix;nlohmann::json metadata;
 ~ActualCaptureFrame5164(){try{std::ofstream f(prefix+"-manifest.json");f<<metadata.dump(2);}catch(...){}}
};
std::unique_ptr<ActualCaptureFrame5164> actual_capture_begin5164(rt::Runtime&r,size_t tokens,size_t offset){
 int d=r.device().index();if(d<0||d>=8||tokens<512)return{};auto&s=actual_capture_states5164[d];
 if(!s.armed.load(std::memory_order_acquire))return{};
 std::lock_guard<std::mutex>lock(s.mutex);if(!s.armed.exchange(false))return{};
 auto out=std::make_unique<ActualCaptureFrame5164>();out->prefix=s.prefix;out->metadata={{"scope","Actual immutable Model first eligible sparse prefill. Last two token rows / sixteen routed rows plus bounded full-prefill route IDs. No weights, no retained GPU tensors, no GPU F64; diagnostic copies excluded from qualified timing."},{"device",d},{"tokens",tokens},{"routes",tokens*8},{"routed_offset",offset},{"fields",nlohmann::json::array()}};return out;
}
void actual_capture_field5164(rt::Runtime&r,ActualCaptureFrame5164*cap,const char*name,const rt::Tensor&t,size_t rows){
 if(!cap)return;
 if(!t||!t->is_contiguous()||!t->ndim()||!t->size(0)){cap->metadata["errors"].push_back(name);return;}
 rows=std::min(rows,t->size(0));size_t stride=t->nbytes()/t->size(0),bytes=rows*stride,start=t->size(0)-rows;
 // A single bounded host vector; no cache or GPU allocation survives this call.
 r.syncStream();std::vector<unsigned char>host(bytes);r.memcpyD2H(host.data(),t->data()+start*stride,bytes);
 std::string file=cap->prefix+"-"+name+".bin";
 if(std::ifstream(file).good()){cap->metadata["errors"].push_back(std::string("exists:")+name);return;}
 std::ofstream f(file,std::ios::binary);f.write(reinterpret_cast<const char*>(host.data()),host.size());
 std::vector<size_t>shape;for(size_t i=0;i<t->ndim();++i)shape.push_back(t->size(i));
 cap->metadata["fields"].push_back({{"field",name},{"file",file},{"full_shape",shape},{"sample_rows",rows},{"start_row",start},{"dtype_enum",int(t->dtype())},{"bytes",bytes},{"write_ok",bool(f)}});
}

struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;int entered_exceptions=std::uncaught_exceptions();~Work(){decode_leases_r1354::retire(r,held,entered_exceptions);}rt::Tensor alloc(rt::Shape shape,rt::DataType type){auto x=rt::Tensor::empty(r,shape,type);held.push_back(x);return x;}rt::Tensor ids(const std::vector<int64_t>&v){auto x=alloc({v.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),v.data(),v.size()*8,false);return x;}};
}
Fp8MlpR19::Fp8MlpR19(rt::Runtime&r,size_t intermediate,const engine::distributed::RankInfo&rank):owner_(&r),local_(0),global_(intermediate),offset_(0){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(rank.tp_size!=1&&rank.tp_size!=8)||rank.tp_rank<0||rank.tp_rank>=rank.tp_size||!intermediate||intermediate%size_t(rank.tp_size)||intermediate/size_t(rank.tp_size)%128)throw std::invalid_argument("GLM MLP requires block-aligned TP1/8 partition");
 local_=intermediate/size_t(rank.tp_size);offset_=size_t(rank.tp_rank)*local_;
 try{gate_=bind_child("gate_proj",std::make_shared<Fp8Projection>(r,intermediate,6144,rank.tp_size,rank.tp_rank));up_=bind_child("up_proj",std::make_shared<Fp8Projection>(r,intermediate,6144,rank.tp_size,rank.tp_rank));down_=rt::Tensor::empty_parameter(r,{6144,local_},rt::DataType::kUInt8);scale_=rt::Tensor::empty_parameter(r,{48,global_/128},rt::DataType::kFloat32);bind_weight("down_proj.weight",weights::Weight(down_,1,rank.tp_rank,rank.tp_size));bind_weight("down_proj.weight_scale_inv",weights::Weight(scale_,-1,rank.tp_rank,rank.tp_size));r.syncStream();}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void Fp8MlpR19::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::logic_error("GLM MLP postload differs");r.syncStream();ready_=true;}
rt::Tensor Fp8MlpR19::forward_local(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();if(&r!=owner_||!ready_||r.isGraphRecording()||!x||x->device()!=r.device()||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=6144||!x->is_contiguous())throw std::invalid_argument("GLM MLP requires loaded contiguous BF16 hidden");
 Work work{r};work.held.push_back(x);size_t tokens=x->size(0);auto gate=gate_->forward(r,x),up=up_->forward(r,x);work.held.insert(work.held.end(),{gate,up});auto silu=work.alloc({tokens,local_},rt::DataType::kBFloat16),activated=work.alloc({tokens,local_},rt::DataType::kBFloat16);
 // Preserve the BF16 activation boundary used by the eager checkpoint reference.
 ex::submit<infini::ops::Silu>(r,infini::ops::Silu::DefaultConfig(r.device().type()),gate,silu);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),silu,up,activated);
 auto packed=work.alloc(activated->shape(),rt::DataType::kUInt8),local_scale=work.alloc({tokens,local_/128},rt::DataType::kFloat32),global_scale=work.alloc({tokens,global_/128},rt::DataType::kFloat32),out=work.alloc({tokens,6144},rt::DataType::kBFloat16);
 ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),activated,packed,local_scale);r.setDeviceMemory(global_scale->data(),0xff,global_scale->nbytes());global_scale->narrow({{1,offset_/128,local_/128}})->copy_from(r,local_scale);
 ex::submit<infini::ops::Fp8BlockLinear>(r,ex::RecordingPolicy::HostOnly,infini::ops::Fp8BlockLinear::DefaultConfig(r.device().type()),out,packed,global_scale,down_,scale_,size_t{0},offset_);decode_leases_r1354::success_fence(r);return out;
}
FeedForwardR19::FeedForwardR19(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank):owner_(&r),sparse_(cfg.at("mlp_layer_types").at(layer)=="sparse"),tp_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(tp_!=1&&tp_!=8)||(tp_>1&&!comm_)||cfg.at("hidden_size")!=6144||cfg.at("hidden_act")!="silu"||(cfg.at("mlp_layer_types").at(layer)!="dense"&&!sparse_)||cfg.at("intermediate_size")!=12288||cfg.at("moe_intermediate_size")!=2048||cfg.at("n_routed_experts")!=256||cfg.at("num_experts_per_tok")!=8||cfg.at("n_shared_experts")!=1||cfg.at("n_group")!=1||cfg.at("topk_group")!=1||cfg.at("scoring_func")!="sigmoid"||!cfg.at("norm_topk_prob").get<bool>())throw std::invalid_argument("GLM FFN actual checkpoint configuration differs");
 try{auto alias=[&](const std::string&prefix,std::shared_ptr<Fp8MlpR19>module){for(auto&item:module->state_dict())bind_weight(prefix+item.first,item.second);return module;};
  if(!sparse_)dense_=alias("",std::make_shared<Fp8MlpR19>(r,12288,rank));
  else{gate_=rt::Tensor::empty_parameter(r,{256,6144},rt::DataType::kBFloat16);gate_float_=rt::Tensor::empty(r,{256,6144},rt::DataType::kFloat32);bias_=rt::Tensor::empty_parameter(r,{256},rt::DataType::kFloat32);bind_weight("gate.weight",weights::Weight(gate_));bind_weight("gate.e_score_correction_bias",weights::Weight(bias_));if(tp_!=8)throw std::invalid_argument("GPU-batched sparse GLM candidate requires TP8");
  routed_offset_=size_t(rank.tp_rank)*256;
  routed_gate_=rt::Tensor::empty_parameter(r,{256,256,6144},rt::DataType::kUInt8);
  routed_up_=rt::Tensor::empty_parameter(r,{256,256,6144},rt::DataType::kUInt8);
  routed_down_=rt::Tensor::empty_parameter(r,{256,6144,256},rt::DataType::kUInt8);
  routed_gate_scale_=rt::Tensor::empty_parameter(r,{256,16,48},rt::DataType::kFloat32);
  routed_up_scale_=rt::Tensor::empty_parameter(r,{256,16,48},rt::DataType::kFloat32);
  routed_down_scale_=rt::Tensor::empty_parameter(r,{256,48,16},rt::DataType::kFloat32);
  for(size_t i=0;i<256;++i){
   const auto prefix="experts."+std::to_string(i)+".";
   auto slice=[&](const rt::Tensor&t,rt::Shape shape){return t->narrow({{0,i,1}})->view(shape);};
   bind_weight(prefix+"gate_proj.weight",weights::Weight(slice(routed_gate_,{256,6144}),0,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"up_proj.weight",weights::Weight(slice(routed_up_,{256,6144}),0,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"down_proj.weight",weights::Weight(slice(routed_down_,{6144,256}),1,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"gate_proj.weight_scale_inv",weights::Weight(slice(routed_gate_scale_,{16,48}),-1,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"up_proj.weight_scale_inv",weights::Weight(slice(routed_up_scale_,{16,48}),-1,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"down_proj.weight_scale_inv",weights::Weight(slice(routed_down_scale_,{48,16}),-1,rank.tp_rank,rank.tp_size));
  }shared_=alias("shared_experts.",std::make_shared<Fp8MlpR19>(r,2048,rank));float eps=1e-20f,factor=cfg.at("routed_scaling_factor").get<float>();if(!std::isfinite(factor)||factor<=0)throw std::invalid_argument("GLM FFN invalid routed scale");epsilon_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);factor_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);r.memcpyH2D(epsilon_->data(),&eps,4,false);r.memcpyH2D(factor_->data(),&factor,4,false);}
  r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void FeedForwardR19::process_weights_after_loading(rt::Runtime&r,bool transpose){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::logic_error("GLM FFN postload differs");if(dense_)dense_->process_weights_after_loading(r,transpose);else{for(auto&expert:experts_)expert->process_weights_after_loading(r,transpose);shared_->process_weights_after_loading(r,transpose);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),gate_,gate_float_);}r.syncStream();ready_=true;}
FeedForwardR19::Output FeedForwardR19::forward(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();if(&r!=owner_||!ready_||r.isGraphRecording()||!x||x->device()!=r.device()||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->is_contiguous()||!x->size(0)||x->size(0)>4096||x->size(1)!=6144)throw std::invalid_argument("GLM FFN requires loaded BF16 packed hidden");
 glm_vector_r8540::Frame vector_frame_r8540(r.device().index());Work work{r};work.held.push_back(x);size_t tokens=x->size(0);std::unique_ptr<ActualCaptureFrame5164>capture;auto combined=work.alloc({tokens,6144},rt::DataType::kFloat32);rt::Tensor indices,weights;
 if(dense_){auto local=dense_->forward_local(r,x);work.held.push_back(local);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),local,combined);}
 else{capture=actual_capture_begin5164(r,tokens,routed_offset_);auto input=work.alloc(x->shape(),rt::DataType::kFloat32),logits=work.alloc({tokens,256},rt::DataType::kFloat32),scores=work.alloc({tokens,256},rt::DataType::kFloat32),choice=work.alloc({tokens,256},rt::DataType::kFloat32),chosen=work.alloc({tokens,8},rt::DataType::kFloat32);indices=work.alloc({tokens,8},rt::DataType::kInt64);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),x,input);ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),input,gate_float_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits);ex::submit<infini::ops::Sigmoid>(r,infini::ops::Sigmoid::DefaultConfig(r.device().type()),logits,scores);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(r.device().type()),scores,bias_,choice);ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(r.device().type()),choice,int64_t{8},int64_t{1},true,true,chosen,indices);
  auto selected=work.alloc({tokens,8},rt::DataType::kFloat32),sum=work.alloc({tokens,1},rt::DataType::kFloat32),denominator=work.alloc({tokens,1},rt::DataType::kFloat32),normalized=work.alloc({tokens,8},rt::DataType::kFloat32);weights=work.alloc({tokens,8},rt::DataType::kFloat32);
  ex::submit<infini::ops::Gather>(r,infini::ops::Gather::DefaultConfig(r.device().type()),scores,indices,int64_t{1},false,selected);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),selected,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},sum);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(r.device().type()),sum,epsilon_,denominator);ex::submit<infini::ops::Div>(r,infini::ops::Div::DefaultConfig(r.device().type()),selected,denominator,normalized);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),normalized,factor_,weights);
  // indices are produced immediately above by public Topk over 256 experts;
  // their range is established by that operator, not unchecked user metadata.
  // All banks alias the original checkpoint names; no duplicate expert weights.
  glm_layer_mark_r5670(r,"FFN.router.end");const size_t routes=tokens*8;
  auto packed=work.alloc({tokens,6144},rt::DataType::kUInt8),input_scale=work.alloc({tokens,48},rt::DataType::kFloat32);
  ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),x,packed,input_scale);
  actual_capture_field5164(r,capture.get(),"hidden_BF16",x,2);actual_capture_field5164(r,capture.get(),"packed_FP8",packed,2);actual_capture_field5164(r,capture.get(),"input_scale_F32",input_scale,2);actual_capture_field5164(r,capture.get(),"route_ids_I64",indices,tokens);actual_capture_field5164(r,capture.get(),"route_weights_F32",weights,2);
   auto& diagnostic=routed_ceiling9046;
   const bool eligible=tokens==8 && routes==64 && routed_offset_%256==0;
   rt::Tensor local;
   if(eligible){diagnostic.current_x=x;diagnostic.current_ids=indices;}
   if(eligible && diagnostic.mode==1) {
    if(!diagnostic.local || diagnostic.owner!=&r || diagnostic.module!=this || diagnostic.offset!=routed_offset_)throw std::logic_error("routed ceiling fixture not captured");
    local=diagnostic.local;work.held.push_back(local);++diagnostic.counts[1];
   } else {
    local=work.alloc({routes,6144},rt::DataType::kBFloat16);
  glm_layer_mark_r5670(r,"FFN.gateup.begin");auto gates=work.alloc({routes,256},rt::DataType::kBFloat16),ups=work.alloc({routes,256},rt::DataType::kBFloat16);
  if(glm_vector_r8540::gate(vector_frame_r8540,gates->data(),ups->data(),packed->data(),input_scale->data(),indices->data(),routed_gate_->data(),routed_gate_scale_->data(),routed_up_->data(),routed_up_scale_->data(),routes,routed_offset_,r.stream()))throw std::runtime_error("GLM routed gate/up launch failed");
  actual_capture_field5164(r,capture.get(),"gate_BF16",gates,16);actual_capture_field5164(r,capture.get(),"up_BF16",ups,16);
  glm_layer_mark_r5670(r,"FFN.gateup.end");auto silu=work.alloc({routes,256},rt::DataType::kBFloat16),activated=work.alloc({routes,256},rt::DataType::kBFloat16);
  ex::submit<infini::ops::Silu>(r,infini::ops::Silu::DefaultConfig(r.device().type()),gates,silu);
  ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),silu,ups,activated);
  glm_layer_mark_r5670(r,"FFN.downquant.begin");auto down_input=work.alloc({routes,256},rt::DataType::kUInt8),down_scale=work.alloc({routes,2},rt::DataType::kFloat32);
  ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),activated,down_input,down_scale);
  actual_capture_field5164(r,capture.get(),"activated_BF16",activated,16);actual_capture_field5164(r,capture.get(),"down_input_FP8",down_input,16);actual_capture_field5164(r,capture.get(),"down_scale_F32",down_scale,16);
  glm_layer_mark_r5670(r,"FFN.down.begin");if(glm_vector_r8540::down(vector_frame_r8540,local->data(),down_input->data(),down_scale->data(),indices->data(),routed_down_->data(),routed_down_scale_->data(),routes,routed_offset_,r.stream()))throw std::runtime_error("GLM routed down launch failed");
    if(eligible) {
     if(diagnostic.mode==2) { diagnostic.local=local;diagnostic.owner=&r;diagnostic.module=this;diagnostic.offset=routed_offset_;++diagnostic.counts[2]; }
     else ++diagnostic.counts[0];
    }
   }
  actual_capture_field5164(r,capture.get(),"local_down_BF16",local,16);
  glm_layer_mark_r5670(r,"FFN.down.end");auto local_float=work.alloc({routes,6144},rt::DataType::kFloat32),contributions=work.alloc({routes,6144},rt::DataType::kFloat32);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),local,local_float);
  ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),local_float,weights->view({routes,1}),contributions);
  actual_capture_field5164(r,capture.get(),"weighted_contributions_F32",contributions,16);
  glm_layer_mark_r5670(r,"FFN.shared.begin");auto routed=work.alloc({tokens,6144},rt::DataType::kFloat32),shared=shared_->forward_local(r,x),shared_float=work.alloc({tokens,6144},rt::DataType::kFloat32);work.held.push_back(shared);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),contributions->view({tokens,8,6144}),std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},routed);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),shared,shared_float);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(r.device().type()),routed,shared_float,combined);
  actual_capture_field5164(r,capture.get(),"routed_reduce_F32",routed,2);actual_capture_field5164(r,capture.get(),"shared_BF16",shared,2);actual_capture_field5164(r,capture.get(),"combined_preTP_F32",combined,2);
 }
 glm_layer_mark_r5670(r,"FFN.shared.end_TP.begin");auto reduced=combined;if(tp_>1){reduced=work.alloc(combined->shape(),combined->dtype());reduced->copy_from(r,combined);engine::distributed::allreduce_(r,reduced,reduced,infinicclSum,comm_);}glm_layer_mark_r5670(r,"FFN.TP.end_cast.begin");auto out=work.alloc({tokens,6144},rt::DataType::kBFloat16);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),reduced,out);decode_leases_r1354::success_fence(r);actual_capture_field5164(r,capture.get(),"combined_postTP_F32",reduced,2);actual_capture_field5164(r,capture.get(),"hidden_out_BF16",out,2);return {out,combined,weights,indices};
}
}

extern "C" int glm_actual_capture_arm_r5164(int device,const char*prefix){return infinilm::models::glm_moe_dsa::actual_capture_arm5164(device,prefix);}

extern "C" int glm_routed_ceiling_policy_r9046(int mode) {
 using namespace infinilm::models::glm_moe_dsa;
 if(mode==-1){routed_ceiling9046.local.reset();routed_ceiling9046.current_x.reset();routed_ceiling9046.current_ids.reset();routed_ceiling9046.reference_x.clear();routed_ceiling9046.reference_ids.clear();routed_ceiling9046.owner=nullptr;routed_ceiling9046.module=nullptr;routed_ceiling9046.mode=0;return 0;}
 if(mode<0||mode>2)return -1;routed_ceiling9046.mode=mode;return 0;
}
extern "C" int glm_routed_ceiling_counts_r9046(uint64_t*out,size_t n) {
 using namespace infinilm::models::glm_moe_dsa;if(!out||n!=3)return -1;
 for(size_t i=0;i<3;++i)out[i]=routed_ceiling9046.counts[i];return 0;
}
extern "C" int glm_routed_ceiling_validate_r9046(uint64_t*out,size_t n) {
 using namespace infinilm::models::glm_moe_dsa;auto&s=routed_ceiling9046;
 if(!out||n!=3||!s.owner||!s.current_x||!s.current_ids||!s.local)return -1;
 try { s.owner->assertOwnerThread();std::vector<unsigned char>x(s.current_x->nbytes()),ids(s.current_ids->nbytes());
 s.owner->memcpyD2H(x.data(),s.current_x->data(),x.size());s.owner->memcpyD2H(ids.data(),s.current_ids->data(),ids.size());
 if(s.reference_x.empty()){s.reference_x=x;s.reference_ids=ids;}
 out[0]=x==s.reference_x;out[1]=ids==s.reference_ids;out[2]=x.size()+ids.size();return 0;
 } catch(...){return -2;}
}
