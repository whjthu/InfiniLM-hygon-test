#include "glm_layer_trace_r5670.hpp"
#include "glm_exact_banks_r4188.hpp"
#include "glm_moe_dsa_projected_cache_r22.hpp"
#include "glm_selection_map_r302.hpp"
#include "glm_moe_dsa_indexer_rotary.hpp"
#include "glm_moe_dsa_sparse_attention.hpp"
#include "execution/submit.hpp"
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <array>
#include <atomic>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8>view_calls927{},copy_calls927{},proven_prefix_calls1669{},trusted_decode_calls1669{};}
extern "C" uint64_t modern_glm_cache_view_calls_r927(int device,int kind){if(device<0||device>=8||kind<0||kind>1)return 0;return kind?copy_calls927[size_t(device)].load():view_calls927[size_t(device)].load();}
namespace {thread_local std::map<const rt::Runtime*,bool> selection_policy998;}
void glm_indexer_policy_r998(rt::Runtime&r,bool enabled){r.assertOwnerThread();if(r.isGraphRecording())throw std::invalid_argument("998 policy recording");r.syncStream();selection_policy998[&r]=enabled;}
void glm_indexer_policy_clear_r998(rt::Runtime&r){r.assertOwnerThread();r.syncStream();selection_policy998.erase(&r);}
rt::Tensor sparse_attention_exact_rows_r5555(rt::Runtime&,const rt::Tensor&,const std::vector<rt::Tensor>&,const std::vector<rt::Tensor>&,int);
rt::Tensor sparse_attention_all_selected_decode_r601(rt::Runtime&,const rt::Tensor&,const rt::Tensor&,const rt::Tensor&,float);
rt::Tensor sparse_attention_all_selected_decode_batch_r5096(rt::Runtime&,const rt::Tensor&,const rt::Tensor&,const rt::Tensor&);
namespace {
struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}rt::Tensor ids(const std::vector<int64_t>&v){auto x=alloc({v.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),v.data(),v.size()*8,false);return x;}};
}
NativeAttentionProjectedR22::NativeAttentionProjectedR22(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank,size_t rows,size_t capacity,size_t topk,const bool*bounded_policy,const bool*owned_prefix_policy):owner_(&r),bounded_policy_r1406_(bounded_policy),owned_prefix_policy_r1669_(owned_prefix_policy),rows_(rows),capacity_(capacity),maximum_(cfg.at("max_position_embeddings")),topk_(topk),full_(cfg.at("indexer_types").at(layer)=="full"){
 r.assertOwnerThread();if(r.isGraphRecording()||!rows||rows>128||!capacity||capacity>maximum_||!topk||topk>2048||(cfg.at("indexer_types").at(layer)!="full"&&cfg.at("indexer_types").at(layer)!="shared"))throw std::invalid_argument("GLM persistent attention configuration differs");
 try{prepare_=std::make_shared<AttentionPrepare>(r,cfg,rank);output_=std::make_shared<AttentionOutput>(r,cfg,rank);for(auto&item:prepare_->state_dict())bind_weight(item.first,item.second);for(auto&item:output_->state_dict())bind_weight(item.first,item.second);
  if(full_){indexer_=std::make_shared<NativeIndexer>(r,cfg,rank);for(auto&item:indexer_->state_dict())bind_weight("indexer."+item.first,item.second);selector_=std::make_unique<IndexerSelectionR1581>(r,rows,capacity,32,128,topk,rank);r.setDeviceMemory(selector_->cache()->data(),0xff,selector_->cache()->nbytes());}
  latent_=exact_banks_r4188::allocate(r,{rows,capacity,512});rotary_=exact_banks_r4188::allocate(r,{rows,capacity,64});r.setDeviceMemory(latent_->data(),0xff,latent_->nbytes());r.setDeviceMemory(rotary_->data(),0xff,rotary_->nbytes());// Keep the original compressed cache contract and add model-owned projected
  // BF16 K/V banks. Only validated prefix lengths authorize reading a bank row.
  const bool host4504=projected_host_r4504::enabled();
  if(host4504)host_bank_r4504_=std::make_unique<projected_host_r4504::Bank>(r,rows,capacity,prepare_->local_heads());
  projected_key_=exact_banks_r4188::allocate(r,{host4504?size_t{1}:rows,capacity,prepare_->local_heads(),256});
  projected_value_=exact_banks_r4188::allocate(r,{host4504?size_t{1}:rows,capacity,prepare_->local_heads(),256});
  r.setDeviceMemory(projected_key_->data(),0xff,projected_key_->nbytes());
  r.setDeviceMemory(projected_value_->data(),0xff,projected_value_->nbytes());
  lengths_.assign(rows,0);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void NativeAttentionProjectedR22::process_weights_after_loading(rt::Runtime&r,bool flag){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::invalid_argument("GLM persistent attention postload differs");prepare_->process_weights_after_loading(r,flag);output_->process_weights_after_loading(r,flag);if(indexer_)indexer_->process_weights_after_loading(r,flag);ready_=true;}
void NativeAttentionProjectedR22::reset(rt::Runtime&r){r.assertOwnerThread();if(&r!=owner_||r.isGraphRecording())throw std::invalid_argument("GLM persistent attention reset owner differs");r.syncStream();if(selector_)selector_->reset(r);if(host_bank_r4504_)host_bank_r4504_->reset();std::fill(lengths_.begin(),lengths_.end(),0);resident_slots_r5640_.reset();poisoned_=false;}
NativeAttentionProjectedR22::Output NativeAttentionProjectedR22::forward(rt::Runtime&r,const rt::Tensor&hidden,const rt::Tensor&positions,const std::vector<Request>&requests,const std::vector<rt::Tensor>&shared,const PrefixProofsR1645&shared_proofs){
 r.assertOwnerThread();if(&r!=owner_||!ready_||poisoned_||r.isGraphRecording()||!hidden||hidden->device()!=r.device()||hidden->dtype()!=rt::DataType::kBFloat16||hidden->ndim()!=2||!hidden->is_contiguous()||!hidden->size(0)||hidden->size(0)>4096||hidden->size(1)!=6144||!positions||positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||!positions->is_contiguous()||positions->shape()!=rt::Shape{hidden->size(0)}||requests.empty()||(full_?!shared.empty():shared.size()!=requests.size()))throw std::invalid_argument("GLM persistent attention input/selection differs");
 size_t tokens=hidden->size(0),total=0;std::set<size_t>seen;std::vector<int64_t>pos(tokens);r.memcpyD2H(pos.data(),positions->data(),tokens*8);
 // Reject all external metadata before either indexer or latent cache mutates.
 for(size_t i=0;i<requests.size();++i){auto request=requests[i];if(request.row>=rows_||!seen.insert(request.row).second||!request.tokens||request.past!=lengths_[request.row]||request.past>capacity_||request.tokens>capacity_-request.past||request.tokens>tokens-total)throw std::invalid_argument("GLM persistent attention request row/past/count differs");for(size_t t=0;t<request.tokens;++t)if(pos[total+t]!=int64_t(request.past+t))throw std::invalid_argument("GLM persistent attention positions differ");
  if(!full_){auto ids=shared[i];if(!ids||ids->device()!=r.device()||ids->dtype()!=rt::DataType::kInt64||!ids->is_contiguous()||ids->ndim()!=2||ids->size(0)!=request.tokens||!ids->size(1)||ids->size(1)>topk_)throw std::invalid_argument("GLM shared indexer selection shape differs");std::vector<int64_t>host(ids->numel());r.memcpyD2H(host.data(),ids->data(),host.size()*8);for(size_t t=0;t<request.tokens;++t){bool causal=false;for(size_t j=0;j<ids->size(1);++j){auto id=host[t*ids->size(1)+j];if(id<0||size_t(id)>=request.past+request.tokens)throw std::invalid_argument("GLM shared selection outside valid prefix");causal|=size_t(id)<=request.past+t;}if(!causal)throw std::invalid_argument("GLM shared selection has no causal entry");}}
  total+=request.tokens;
 }
 if(total!=tokens)throw std::invalid_argument("GLM persistent attention packed count differs");
 const bool resident5640=resident_policy_r5640_&&*resident_policy_r5640_;
 glm_model_resident_r5640::Slots resident_next5640=resident_slots_r5640_;
 if(resident5640){resident_validate_r5640(r,requests);std::vector<glm_model_resident_r5640::Request>q;for(auto x:requests)q.push_back({x.row,x.past,x.tokens});resident_next5640=resident_slots_r5640_.propose(q);}
 const char*batch_flag5055=std::getenv("GLM_BATCH_ALLSELECTED_R5096");if(batch_flag5055&&std::string(batch_flag5055)!="0"&&std::string(batch_flag5055)!="1")throw std::invalid_argument("5055 batch attention flag 0/1");
 const bool request_batch5055=batch_flag5055&&std::string(batch_flag5055)=="1";
 Work work{r};try{glm_layer_mark_r5670(r,"attn.prepare.begin");auto prepared=prepare_->forward_latent(r,hidden,positions);work.held.insert(work.held.end(),{prepared.query,prepared.q_resid,prepared.kv_latent,prepared.k_rot});std::vector<rt::Tensor>selection;PrefixProofsR1645 selection_proofs;
glm_layer_mark_r5670(r,"attn.indexer.begin");  if(full_){bool all_visible=selection_policy998.count(&r)&&selection_policy998.at(&r);
   for(auto request:requests)all_visible=all_visible&&request.tokens==1&&request.past<topk_;
   bool metadata_only=capacity_<=topk_&&bounded_policy_r1406_&&*bounded_policy_r1406_;
   for(auto request:requests)metadata_only=metadata_only&&request.tokens==1&&request.past<topk_;
   if(metadata_only){
    if(owned_prefix_policy_r1669_&&*owned_prefix_policy_r1669_){auto proven=selector_->select_all_metadata_proven_readonly(r,requests);selection=std::move(proven.values);selection_proofs=std::move(proven.proofs);proven_prefix_calls1669.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);}
    else selection=selector_->select_all_metadata(r,requests);
    work.held.insert(work.held.end(),selection.begin(),selection.end());}
   else{auto features=indexer_->forward(r,hidden,prepared.q_resid,positions);work.held.insert(work.held.end(),{features.query,features.key,features.head_weights});
    if(all_visible){selection=selector_->select_all(r,features.key,requests);work.held.insert(work.held.end(),selection.begin(),selection.end());}
    else{auto results=selector_->forward(r,features.query,features.key,features.head_weights,requests);for(auto&result:results){work.held.insert(work.held.end(),{result.scores,result.indices});selection.push_back(result.indices);}}}}
  else{selection=shared;selection_proofs=shared_proofs;}
  // Project independent arriving KV rows once for the complete packed batch.
  // Request validation and selection stay above; per-request cache writes,
  // sparse selection/order and length publication remain unchanged.
glm_layer_mark_r5670(r,"attn.projectedKV.begin");  auto fresh_all=prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot);
  work.held.insert(work.held.end(),{fresh_all.first,fresh_all.second});
glm_layer_mark_r5670(r,"attn.cache.begin");  const char*group_flag=std::getenv("GLM_GROUP_PROJECTED_HOST_R5030");
  if(group_flag&&std::string(group_flag)!="0"&&std::string(group_flag)!="1")throw std::invalid_argument("5026 group flag 0/1");
  const bool group5026=!resident5640&&host_bank_r4504_&&group_flag&&std::string(group_flag)=="1"&&requests.size()>=8;
  if(!resident5640&&host_bank_r4504_)host_bank_r4504_->drain();
  if(group5026){
   std::vector<size_t>rr,pp,cc;for(auto request:requests){rr.push_back(request.row);pp.push_back(request.past);cc.push_back(request.tokens);}
   // Already packed native projection tensors: no staging or clone required.
   host_bank_r4504_->write_group(rr,pp,cc,fresh_all.first,fresh_all.second,static_cast<cudaStream_t>(r.stream()));
  }
  bool batch5055=!resident5640&&request_batch5055&&host_bank_r4504_&&requests.size()>=8&&requests.size()<=128&&tokens==requests.size()&&requests[0].past+1<=640&&owned_prefix_policy_r1669_&&*owned_prefix_policy_r1669_&&bounded_policy_r1406_&&*bounded_policy_r1406_&&capacity_<=topk_&&selection_proofs.size()==requests.size();
  for(size_t i=0;batch5055&&i<requests.size();++i){const auto&request=requests[i];batch5055=request.tokens==1&&request.past==requests[0].past&&selection_proofs[i]&&selection_proofs[i]->matches(r,selection[i],request.row,request.past,1,capacity_);}
  rt::Tensor batch_key5055,batch_value5055;
  if(batch5055){batch_key5055=work.alloc({requests.size(),requests[0].past+1,prepare_->local_heads(),256},rt::DataType::kBFloat16);batch_value5055=work.alloc(batch_key5055->shape(),rt::DataType::kBFloat16);}
  bool shared5640=resident5640&&requests.size()==8&&tokens==8&&owned_prefix_policy_r1669_&&*owned_prefix_policy_r1669_&&bounded_policy_r1406_&&*bounded_policy_r1406_&&capacity_<=topk_&&selection_proofs.size()==requests.size();
  for(size_t i=0;shared5640&&i<requests.size();++i){auto q=requests[i];shared5640=q.tokens==1&&selection_proofs[i]&&selection_proofs[i]->matches(r,selection[i],q.row,q.past,1,capacity_);}
  std::vector<rt::Tensor>resident_keys5640,resident_values5640;
  // All Model and perlayer slot metadata is validated BEFORE these appends.
  // Leading prefix of one slot is truly contiguous; no640-stride paddedview.
  if(resident5640){size_t offset=0;work.held.insert(work.held.end(),{resident_key_r5640_,resident_value_r5640_});
   for(auto q:requests){size_t slot=size_t(resident_next5640.slot(q.row));auto kr=resident_key_r5640_->narrow({{0,slot,1}})->view({640,8,256}),vr=resident_value_r5640_->narrow({{0,slot,1}})->view({640,8,256});
    kr->narrow({{0,q.past,q.tokens}})->copy_from(r,fresh_all.first->narrow({{0,offset,q.tokens}}));vr->narrow({{0,q.past,q.tokens}})->copy_from(r,fresh_all.second->narrow({{0,offset,q.tokens}}));offset+=q.tokens;resident_append_bytes_r5640_+=8192*q.tokens;
    work.held.insert(work.held.end(),{kr,vr});}}
  auto attention=work.alloc({tokens,prepare_->local_heads(),256},rt::DataType::kBFloat16);size_t begin=0;
  for(size_t i=0;i<requests.size();++i){auto request=requests[i];size_t count=request.tokens,prefix=request.past+count;auto latent_row=latent_->narrow({{0,request.row,1}})->view({capacity_,512}),rotary_row=rotary_->narrow({{0,request.row,1}})->view({capacity_,64});latent_row->narrow({{0,request.past,count}})->copy_from(r,prepared.kv_latent->narrow({{0,begin,count}}));rotary_row->narrow({{0,request.past,count}})->copy_from(r,prepared.k_rot->narrow({{0,begin,count}})->view({count,64}));
   auto key_row=resident5640?resident_key_r5640_->narrow({{0,size_t(resident_next5640.slot(request.row)),1}})->view({640,8,256}):batch5055?batch_key5055->narrow({{0,i,1}})->view({prefix,prepare_->local_heads(),256}):projected_key_->narrow({{0,host_bank_r4504_?size_t{0}:request.row,1}})->view({capacity_,prepare_->local_heads(),256});
   auto value_row=resident5640?resident_value_r5640_->narrow({{0,size_t(resident_next5640.slot(request.row)),1}})->view({640,8,256}):batch5055?batch_value5055->narrow({{0,i,1}})->view({prefix,prepare_->local_heads(),256}):projected_value_->narrow({{0,host_bank_r4504_?size_t{0}:request.row,1}})->view({capacity_,prepare_->local_heads(),256});
   // Project each arriving latent exactly once, retaining the same FP8
   // activation quantization, BF16 outputs, and rotary assembly as the baseline.
   std::pair<rt::Tensor,rt::Tensor>fresh{fresh_all.first->narrow({{0,begin,count}}),fresh_all.second->narrow({{0,begin,count}})};
   if(resident5640){/* All fresh projected KV appended above; hostbank intentionally stale untilreset. */}
   else if(host_bank_r4504_){
    const char*async_flag=std::getenv("GLM_ASYNC_PROJECTED_HOST_R5011");
    const bool async5007=async_flag&&std::string(async_flag)=="1";
    if(async_flag&&std::string(async_flag)!="0"&&std::string(async_flag)!="1")throw std::invalid_argument("5007 async flag 0/1");
    if(group5026){
     if(request.past)host_bank_r4504_->read_past_while_commit(request.row,request.past,key_row,value_row);
     key_row->narrow({{0,request.past,count}})->copy_from(r,fresh.first);
     value_row->narrow({{0,request.past,count}})->copy_from(r,fresh.second);
    }else if(async5007){
     // Drain only this row's previous append before touching its CPU prefix.
     host_bank_r4504_->complete(request.row,true);
     if(request.past)host_bank_r4504_->read(request.row,request.past,key_row,value_row);
     key_row->narrow({{0,request.past,count}})->copy_from(r,fresh.first);
     value_row->narrow({{0,request.past,count}})->copy_from(r,fresh.second);
     // Fresh projected KV tensors remain owned until async CPU commit completes.
     // The current attention consumes GPU data directly; no new-KV H2D roundtrip.
     host_bank_r4504_->write_async(request.row,request.past,count,fresh.first,fresh.second,static_cast<cudaStream_t>(r.stream()));
    }else{
     host_bank_r4504_->write(request.row,request.past,count,fresh.first,fresh.second);
     host_bank_r4504_->read(request.row,prefix,key_row,value_row);
    }
   }else{
    key_row->narrow({{0,request.past,count}})->copy_from(r,fresh.first);
    value_row->narrow({{0,request.past,count}})->copy_from(r,fresh.second);
   }
   if(shared5640){auto kr=key_row->narrow({{0,0,prefix}}),vr=value_row->narrow({{0,0,prefix}});
    if(!kr->is_contiguous()||!vr->is_contiguous()||kr->nbytes()!=4096*prefix||vr->nbytes()!=4096*prefix||reinterpret_cast<uintptr_t>(kr->data())%256||reinterpret_cast<uintptr_t>(vr->data())%256)throw std::logic_error("5640 exactprefix layout");
    resident_keys5640.push_back(kr);resident_values5640.push_back(vr);work.held.insert(work.held.end(),{key_row,value_row,kr,vr});begin+=count;continue;}
   if(batch5055){begin+=count;continue;}
glm_layer_mark_r5670(r,"attn.row.compute.begin");   auto chosen=selection[i];rt::Tensor result;
   const bool certified=owned_prefix_policy_r1669_&&*owned_prefix_policy_r1669_&&bounded_policy_r1406_&&*bounded_policy_r1406_&&capacity_<=topk_&&!selection_proofs.empty();
   if(certified){
    if(selection_proofs.size()!=requests.size()||!selection_proofs[i]||!selection_proofs[i]->matches(r,chosen,request.row,request.past,count,capacity_))throw std::invalid_argument("1669 selection provenance/owner/row/prefix binding differs");
    std::pair<rt::Tensor,rt::Tensor>expanded{key_row->narrow({{0,0,prefix}}),value_row->narrow({{0,0,prefix}})};work.held.insert(work.held.end(),{expanded.first,expanded.second});
    view_calls927.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);trusted_decode_calls1669.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);
    result=sparse_attention_all_selected_decode_r601(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,1.f/16);
   }else{
std::vector<int64_t>host(chosen->numel());r.memcpyD2H(host.data(),chosen->data(),host.size()*8);auto remapped=selection_map_r302(host,count,chosen->size(1),request.past);auto&logical=remapped.logical;auto&mapped=remapped.mapped;
   auto ids=work.ids(logical);
   // Validated source-ordered selection_map output is the complete valid
   // prefix only if every logical position is present exactly once. Borrow
   // immutable contiguous cache views in that case; never expose spare slots.
   bool complete_prefix=logical.size()==prefix;
   for(size_t j=0;complete_prefix&&j<logical.size();++j)complete_prefix=logical[j]==int64_t(j);
   std::pair<rt::Tensor,rt::Tensor> expanded;
   if(complete_prefix){
    expanded={key_row->narrow({{0,0,prefix}}),value_row->narrow({{0,0,prefix}})};
    work.held.insert(work.held.end(),{expanded.first,expanded.second});
    view_calls927.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);
   }else{
    expanded={work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16),work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16)};
    ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),key_row,ids,int64_t{0},expanded.first);
    ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),value_row,ids,int64_t{0},expanded.second);
    copy_calls927.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);
   }

   if(count==1){result=sparse_attention_all_selected_decode_r601(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,1.f/16);}
   else{auto mapped_ids=work.ids(mapped)->view(chosen->shape());result=sparse_attention_mapped_bounded(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16);}
   }
   if(resident5640&&count==1)++resident_row601_calls_r5640_;
   work.held.push_back(result);attention->narrow({{0,begin,count}})->copy_from(r,result);begin+=count;glm_layer_mark_r5670(r,"attn.row.compute.end");
  }
  if(shared5640){auto joined=sparse_attention_exact_rows_r5555(r,prepared.query,resident_keys5640,resident_values5640,1);work.held.push_back(joined);attention->copy_from(r,joined);++resident_shared_calls_r5640_;}
  if(batch5055){auto joined=sparse_attention_all_selected_decode_batch_r5096(r,prepared.query,batch_key5055,batch_value5055);work.held.push_back(joined);attention->copy_from(r,joined);view_calls927.at(r.device().index()).fetch_add(requests.size(),std::memory_order_relaxed);trusted_decode_calls1669.at(r.device().index()).fetch_add(requests.size(),std::memory_order_relaxed);}
  glm_layer_mark_r5670(r,"attn.output.begin");auto projected=output_->forward(r,attention);glm_layer_mark_r5670(r,"attn.output.end");work.held.insert(work.held.end(),{projected.local,projected.hidden});r.syncStream();for(auto request:requests)lengths_[request.row]=request.past+request.tokens;if(resident5640)resident_slots_r5640_=std::move(resident_next5640);return {projected.hidden,selection,prepared,attention,selection_proofs};
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}

extern "C" uint64_t modern_glm_prefix_calls_r1669(int device,int kind){if(device<0||device>=8||kind<0||kind>1)return 0;return(kind?infinilm::models::glm_moe_dsa::trusted_decode_calls1669:infinilm::models::glm_moe_dsa::proven_prefix_calls1669)[size_t(device)].load(std::memory_order_relaxed);}
