// Private Model members. SAME existing Runtime; no component CAPI/stream.
#include "glm_moe_dsa_projected_text_model_r22.hpp"
#include <cuda_runtime_api.h>
#include <limits>
#include "glm_rows16_policy_r5702.hpp"
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;
namespace {
void eager_owner(rt::Runtime&r,const rt::Runtime*expected){
 r.assertOwnerThread();int current=-1;
 if(&r!=expected||r.isGraphRecording()||cudaGetDevice(&current)!=cudaSuccess||current!=r.device().index())throw std::invalid_argument("5640 owner/currentdevice/eager differs");
 cudaStreamCaptureStatus capture;
 if(cudaStreamIsCapturing(static_cast<cudaStream_t>(r.stream()),&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("5640 external Runtime capture");
}
struct Drain{rt::Runtime&r;~Drain(){try{r.syncStream();}catch(...){std::terminate();}}};
}
NativeAttentionProjectedR22::ResidentBuffers NativeAttentionProjectedR22::resident_prepare_r5640(rt::Runtime&r,bool enabled){
 eager_owner(r,owner_);
 if(!ready_||poisoned_||!resident_slots_r5640_.empty())throw std::invalid_argument("5640 attention policy requires ready empty reset");
 for(size_t n:lengths_)if(n)throw std::invalid_argument("5640 attention history notempty");
 ResidentBuffers pending;Drain hold{r}; // drains BEFORE pending tensors release
 if(!enabled)return pending;
 if(!glm_rows16_r5702::resident_geometry(rows_,capacity_,prepare_->local_heads(),bool(host_bank_r4504_)))throw std::invalid_argument("5640 host4504 rows64 cap640 heads8");
 if(resident_key_r5640_&&resident_value_r5640_){pending={resident_key_r5640_,resident_value_r5640_};return pending;}
 pending.first=rt::Tensor::empty(r,{16,640,8,256},rt::DataType::kBFloat16);
 pending.second=rt::Tensor::empty(r,pending.first->shape(),rt::DataType::kBFloat16);
 // Uninitialized spare capacity is never valid/read: Slots tracks exactprefix.
 return pending;
}
void NativeAttentionProjectedR22::resident_publish_r5640(const bool*policy,ResidentBuffers buffers)noexcept{
 // Owner validated + all layers prepared/drained before publication. No GPU
 // operation or throwing allocation between Context mask and layer commit.
 resident_key_r5640_=std::move(buffers.first);resident_value_r5640_=std::move(buffers.second);
 resident_slots_r5640_.reset();resident_policy_r5640_=policy;
}
void NativeAttentionProjectedR22::resident_validate_r5640(rt::Runtime&r,const std::vector<Request>&requests)const{
 if(!resident_policy_r5640_||!*resident_policy_r5640_)return;
 eager_owner(r,owner_);
 if(!ready_||poisoned_||!resident_key_r5640_||!resident_value_r5640_)throw std::invalid_argument("5640 resident notready/poison");
 for(const auto&t:{resident_key_r5640_,resident_value_r5640_})
  if(t->shape()!=rt::Shape{16,640,8,256}||!t->is_contiguous()||t->dtype()!=rt::DataType::kBFloat16||t->device()!=r.device()||t->nbytes()!=glm_model_resident_r5640::bank_bytes/2||reinterpret_cast<uintptr_t>(t->data())%256)throw std::logic_error("5640 resident bank layout");
 std::vector<glm_model_resident_r5640::Request>q;for(auto x:requests)q.push_back({x.row,x.past,x.tokens});
 auto pending=resident_slots_r5640_.propose(q);(void)pending;
}
nlohmann::json NativeAttentionProjectedR22::resident_report_r5640()const{
 resident_slots_r5640_.owner();
 nlohmann::json q={{"enabled",resident_policy_r5640_&&*resident_policy_r5640_},{"active_rows",resident_slots_r5640_.active()},
  {"capacity",640},{"maxactive",16},{"append_D2D_bytes",resident_append_bytes_r5640_},
  {"exact_shared_calls",resident_shared_calls_r5640_},{"resident_original601_row_calls",resident_row601_calls_r5640_},
  {"packed_prefix_bytes",0},{"Runtime_identity",reinterpret_cast<uintptr_t>(owner_)}};
 if(resident_key_r5640_)q["key_shape"]=resident_key_r5640_->shape();
 if(resident_value_r5640_)q["value_shape"]=resident_value_r5640_->shape();return q;
}
TextModelProjectedR22::ResidentPlan TextModelProjectedR22::resident_prepare_r5640(rt::Runtime&r,const glm_model_resident_r5640::Mask&mask){
 check_owner(r);eager_owner(r,owner_);
 if(!ready_||poisoned_||!all_sequences_empty(r)||rank_.pp_size!=2||rank_.tp_size!=8)throw std::invalid_argument("5640 text policy ready/empty/TP8PP2");
 ResidentPlan pending;Drain hold{r};pending.reserve(layers_.size());
 for(size_t i=0;i<layers_.size();++i)pending.push_back(layers_[i]->resident_prepare_r5640(r,mask[begin_+i]));
 return pending;
}
void TextModelProjectedR22::resident_publish_r5640(const bool*mask,ResidentPlan pending,bool trace)noexcept{
 // All indices/counts originate from the same already-validated module/mask.
 resident_any_r5640_=false;
 for(size_t i=0;i<layers_.size();++i){layers_[i]->resident_publish_r5640(mask+begin_+i,std::move(pending[i]));resident_any_r5640_|=mask[begin_+i];}
 resident_memory_trace_r5640_=trace;resident_min_observed_free_r5640_=UINT64_MAX;resident_memory_samples_r5640_=0;
}
void TextModelProjectedR22::resident_sample_memory_r5640(){
 if(!resident_memory_trace_r5640_)return;
 size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess)throw std::runtime_error("5640 memory observer failed");
 resident_min_observed_free_r5640_=std::min(resident_min_observed_free_r5640_,uint64_t(free));++resident_memory_samples_r5640_;
 // Observation only: this is NOT an allocatorpeak and cannot be added to
 // peroperator timings. Caller performs allrank reserve/adoption gates.
}
nlohmann::json TextModelProjectedR22::resident_report_r5640(rt::Runtime&r)const{
 check_owner(r);nlohmann::json q={{"layer_begin",begin_},{"layer_end",end_},{"any_enabled",resident_any_r5640_},
  {"memory_trace",resident_memory_trace_r5640_},{"memory_samples",resident_memory_samples_r5640_},
  {"min_observed_free",resident_min_observed_free_r5640_},{"observation_is_peak",false},
  {"new_workspace_allowance",glm_model_resident_r5640::new_workspace_allowance},{"reserve_bytes",glm_model_resident_r5640::reserve_bytes},
  {"layers",nlohmann::json::array()},{"Runtime_identity",reinterpret_cast<uintptr_t>(owner_)}};
 for(size_t i=0;i<layers_.size();++i){auto a=layers_[i]->resident_report_r5640();a["global_layer"]=begin_+i;a["stage_local_layer"]=i;q["layers"].push_back(a);}return q;
}
} // namespace infinilm::models::glm_moe_dsa
