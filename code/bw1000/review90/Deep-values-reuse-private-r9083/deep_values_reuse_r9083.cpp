#include <algorithm>
#include "deep_selection_r356.hpp"
#include "deepseek_v4_latent_attention.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <atomic>
#include <array>
#include <optional>
#include <limits>
#include <set>
#include <stdexcept>
extern "C" int deep_packed_gather_r374(void*,const void*,const void*,const void*,size_t,size_t,int64_t,int64_t,int64_t,int64_t,void*);
extern "C" int deep_sink_broadcast_r9040(void*,const void*,size_t,size_t,size_t,void*);
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::atomic<int> reference_scratch_mode_r8953{0};std::array<std::array<std::atomic<uint64_t>,4>,8> reference_scratch_counts_r8954{};}

namespace {
std::atomic<int> packed_batch_mode_r8996b{0};
std::array<std::array<std::atomic<uint64_t>,7>,8> packed_batch_counts_r8996b{};
std::atomic<int> sink_broadcast_mode_r9040{0};
std::array<std::array<std::atomic<uint64_t>,3>,8>sink_broadcast_counts_r9040{};
std::atomic<int> scalar_cast_mode_r9054{0};
std::array<std::array<std::atomic<uint64_t>,3>,8>scalar_cast_counts_r9054{};
std::atomic<int> values_reuse_mode_r9083{0};
std::array<std::array<std::atomic<uint64_t>,2>,8>values_reuse_counts_r9083{};
struct PackedRunR8996b {size_t query_begin,count,n,packed_begin;};
bool packed_runs_r8996b(const std::vector<std::vector<int64_t>>&selected,
 const std::vector<size_t>&offsets,size_t packed_rows,std::vector<PackedRunR8996b>&runs){
 if(selected.size()!=2048||offsets.size()!=2049||offsets.front()!=0||offsets.back()!=packed_rows)return false;
 for(size_t t=0;t<2048;++t)if(offsets[t+1]-offsets[t]!=selected[t].size())return false;
 for(size_t t=0;t<2048;){
  const size_t n=selected[t].size(),limit=(t/512+1)*512;size_t end=t+1;
  if(n>2048)return false;
  while(end<limit&&selected[end].size()==n&&offsets[end]==offsets[t]+(end-t)*n)++end;
  if(n&&end-t>=2)runs.push_back({t,end-t,n,offsets[t]});
  t=end;
 }
 return !runs.empty();
}
void packed_run_submit_r8996b(rt::Runtime&r,const PackedRunR8996b&run,
 const rt::Tensor&qf,const rt::Tensor&packed,const rt::Tensor&sink,const rt::Tensor&output,
 const rt::Tensor&logits_store,const rt::Tensor&probs_store,const rt::Tensor&result_store){
 const size_t R=run.count,H=qf->size(1),N=run.n,D=512;
 if(!R||!N||run.packed_begin>packed->size(0)||R> (packed->size(0)-run.packed_begin)/N)
  throw std::logic_error("packed batch span differs");
 // Original r356 concatenation + original r374 gather, no sorting/unique/as_strided.
 // Positive non-overlapping batch stride=N*512; every row is the exact old slice.
 auto values=packed->narrow({{0,run.packed_begin,R*N}})->view({R,N,D});
 auto qr=qf->narrow({{0,run.query_begin,R}});
 auto logits=logits_store->narrow({{0,0,R*H*(N+1)}})->view({R,H,N+1});
 auto probs=probs_store->narrow({{0,0,R*H*(N+1)}})->view({R,H,N+1});
 auto result=result_store->narrow({{0,0,R*H*D}})->view({R,H,D});
 for(size_t t=0;t<R;++t)
  logits->narrow({{0,t,1}})->view({H,N+1})->narrow({{1,N,1}})->copy_from(r,sink->view({H,1}));
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
  qr,values->permute({0,2,1}),std::optional<rt::Tensor>{},std::optional<float>{1.f/std::sqrt(512.f)},
  std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits->narrow({{2,0,N}}));
 ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),
  logits,int64_t{2},std::optional<infini::ops::DataType>{},probs);
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
  probs->narrow({{2,0,N}}),values,std::optional<rt::Tensor>{},std::optional<float>{1},
  std::optional<float>{0},std::optional<int>{},std::optional<int>{},result);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),
  result,output->narrow({{0,run.query_begin,R}}));
 // No intermediate sync: all backing allocations survive original final/exception drain.
}
void softmax_cast_run_r8996b(rt::Runtime&r,const PackedRunR8996b&run,
 const rt::Tensor&qf,const rt::Tensor&packed,const rt::Tensor&sink,const rt::Tensor&output,
 const rt::Tensor&logits_store,const rt::Tensor&probs_store,const rt::Tensor&result_store){
 const size_t R=run.count,H=qf->size(1),N=run.n;
 auto logits=logits_store->narrow({{0,0,R*H*(N+1)}})->view({R,H,N+1});
 auto probs=probs_store->narrow({{0,0,R*H*(N+1)}})->view({R,H,N+1});
 auto results=result_store->narrow({{0,0,R*H*512}})->view({R,H,512});
 const bool broadcast=sink_broadcast_mode_r9040.load()==1;
 if(broadcast){if(R<2||H!=8||!N)throw std::logic_error("r9040 geometry");if(deep_sink_broadcast_r9040(logits->data(),sink->data(),R,H,N,r.stream()))throw std::runtime_error("r9040 sink launch");auto&c=sink_broadcast_counts_r9040.at(r.device().index());++c[0];c[1].fetch_add(R*H);c[2].fetch_add(R-1);}

 const bool reuse_values=values_reuse_mode_r9083.load()==1;
 std::vector<rt::Tensor> value_slices;if(reuse_values)value_slices.reserve(R);
 for(size_t j=0;j<R;++j){
  size_t t=run.query_begin+j;
  auto l=logits->narrow({{0,j,1}})->view({H,N+1});
  if(!broadcast)l->narrow({{1,N,1}})->copy_from(r,sink->view({H,1}));
  auto values=packed->narrow({{0,run.packed_begin+j*N,N}});
  if(reuse_values)value_slices.push_back(values);
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qf->narrow({{0,t,1}})->view({H,512}),values->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1.f/std::sqrt(512.f)},std::optional<float>{0},std::optional<int>{},std::optional<int>{},l->narrow({{1,0,N}}));
 }
 // Only Softmax row count changes. Linked provider dispatch qualification UNKNOWN.
 ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{2},std::optional<infini::ops::DataType>{},probs);
 for(size_t j=0;j<R;++j){
  auto values=reuse_values?value_slices.at(j):packed->narrow({{0,run.packed_begin+j*N,N}});
  auto probabilities=probs->narrow({{0,j,1}})->view({H,N+1});
  auto result=results->narrow({{0,j,1}})->view({H,512});
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),probabilities->narrow({{1,0,N}}),values,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},result);
 }
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),results,output->narrow({{0,run.query_begin,R}}));
 if(reuse_values){auto&c=values_reuse_counts_r9083.at(r.device().index());++c[0];c[1].fetch_add(R);}
 // Every original QK/PV is 2D with identical shape/strides/alpha/operand order.
}

}
rt::Tensor latent_attention_reference_r681(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&swa,const rt::Tensor&compressed,const rt::Tensor&sink,const rt::Tensor&si,const rt::Tensor&ci,const rt::Tensor&lengths){
 r.assertOwnerThread();
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!q||!swa||!compressed||!sink||!si||!ci||!lengths||q->ndim()!=3||!q->size(0)||!q->size(1)||q->size(2)!=512||q->dtype()!=rt::DataType::kBFloat16||swa->ndim()!=2||compressed->ndim()!=2||swa->size(1)!=512||compressed->size(1)!=512||!swa->size(0)||!compressed->size(0)||swa->dtype()!=q->dtype()||compressed->dtype()!=q->dtype()||sink->shape()!=rt::Shape{q->size(1)}||sink->dtype()!=rt::DataType::kFloat32||si->ndim()!=2||ci->ndim()!=2||si->size(0)!=q->size(0)||ci->size(0)!=q->size(0)||!si->size(1)||!ci->size(1)||si->dtype()!=rt::DataType::kInt64||ci->dtype()!=rt::DataType::kInt64||lengths->shape()!=rt::Shape{q->size(0),2}||lengths->dtype()!=rt::DataType::kInt32)
  throw std::invalid_argument("DeepSeek latent attention requires BF16 D512 KV, F32 sink and explicit selected lengths");
 for(const auto&x:{q,swa,compressed,sink,si,ci,lengths})if(x->device()!=r.device())throw std::invalid_argument("DeepSeek latent attention owner differs");
 if(!si->is_contiguous()||!ci->is_contiguous()||!lengths->is_contiguous())throw std::invalid_argument("DeepSeek selection metadata must be contiguous");
 size_t tokens=q->size(0),heads=q->size(1),sw=si->size(1),cw=ci->size(1);std::vector<int64_t>hs(tokens*sw),hc(tokens*cw);std::vector<int32_t>hl(tokens*2);
 r.memcpyD2H(hs.data(),si->data(),hs.size()*8);r.memcpyD2H(hc.data(),ci->data(),hc.size()*8);r.memcpyD2H(hl.data(),lengths->data(),hl.size()*4);
 auto selection_r356=latent_selection_r356(hs,hc,hl,tokens,sw,cw,swa->size(0),compressed->size(0));
 auto&selected=selection_r356.selected;auto&packed_ids=selection_r356.packed_ids;auto&packed_offsets=selection_r356.packed_offsets;
 const bool packed_requested_r8996b=packed_batch_mode_r8996b.load()!=0;
 std::vector<PackedRunR8996b>runs_r8996b;
 const bool packed_batch_r8996b=packed_requested_r8996b&&reference_scratch_mode_r8953.load()==1&&heads==8&&sw==128&&packed_runs_r8996b(selected,packed_offsets,packed_ids.size(),runs_r8996b);
 if(packed_ids.size()>std::numeric_limits<size_t>::max()/2048)throw std::invalid_argument("DeepSeek packed latent selection size overflows");
 std::vector<rt::Tensor>held;auto alloc=[&](rt::Shape shape,rt::DataType dt=rt::DataType::kFloat32){auto x=rt::Tensor::empty(r,shape,dt);held.push_back(x);return x;};
 try{auto output=alloc({tokens,heads,512},q->dtype());
  auto qf=alloc({tokens,heads,512});ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),q,qf);
  rt::Tensor packed_values;
  if(!packed_ids.empty()){
   auto ids=alloc({packed_ids.size()},rt::DataType::kInt64);r.memcpyH2D(ids->data(),packed_ids.data(),packed_ids.size()*8,true);
   packed_values=alloc({packed_ids.size(),512});
   int status=deep_packed_gather_r374(packed_values->data(),swa->data(),compressed->data(),ids->data(),packed_ids.size(),swa->size(0),swa->stride(0),swa->stride(1),compressed->stride(0),compressed->stride(1),r.stream());
   if(status)throw std::runtime_error("DeepSeek fused packed gather failed: "+std::to_string(status));
  }
  // Proposal only: same-stream scratch, retained in held until original completion drain.
  const bool reuse=reference_scratch_mode_r8953.load()==1&&tokens>64;
  rt::Tensor logits_storage,probability_storage,result_storage;
  if(reuse){size_t max_n=0;for(const auto&v:selected)if(v.size()>max_n)max_n=v.size();logits_storage=alloc({heads*(max_n+1)});probability_storage=alloc({heads*(max_n+1)});result_storage=alloc({heads,512});}
  rt::Tensor batch_logits_r8996b,batch_probs_r8996b,batch_result_r8996b;
  if(packed_batch_r8996b){size_t max_logits=0,max_result=0;for(const auto&v:runs_r8996b){max_logits=std::max(max_logits,v.count*heads*(v.n+1));max_result=std::max(max_result,v.count*heads*512);}batch_logits_r8996b=alloc({max_logits});batch_probs_r8996b=alloc({max_logits});batch_result_r8996b=alloc({max_result});}
  // Default OFF. Prefix intervals stop before every packed run and request boundary.
  const bool scalar_cast_r9054=scalar_cast_mode_r9054.load()==1&&packed_batch_r8996b&&packed_batch_mode_r8996b.load()==2&&sink_broadcast_mode_r9040.load()==1;
  std::vector<size_t>prefix_end_r9054(tokens,0);
  size_t largest_prefix_r9054=0;
  if(scalar_cast_r9054){
   std::vector<unsigned char>batched(tokens,0);for(const auto&run:runs_r8996b)for(size_t j=0;j<run.count;++j)batched[run.query_begin+j]=1;
   for(size_t t=0;t<tokens;){if(batched[t]){++t;continue;}size_t e=t+1,limit=(t/512+1)*512;while(e<limit&&!batched[e])++e;
    // Only the actual current short prefix; isolated tails and other shapes stay original.
    if(e-t>1&&e-t<=127){prefix_end_r9054[t]=e;largest_prefix_r9054=std::max(largest_prefix_r9054,e-t);}t=e;}
  }
  rt::Tensor prefix_results_r9054;if(largest_prefix_r9054)prefix_results_r9054=alloc({largest_prefix_r9054,heads,512});
  size_t prefix_begin_r9054=0,prefix_stop_r9054=0,skipped_casts_r9054=0;
  size_t next_run_r8996b=0,scalar_queries_r8996b=0,batch_queries_r8996b=0,scalar_ops_r8996b=0;
  for(size_t t=0;t<tokens;++t){
   if(packed_batch_r8996b&&next_run_r8996b<runs_r8996b.size()&&t==runs_r8996b[next_run_r8996b].query_begin){
    const auto&run=runs_r8996b[next_run_r8996b];(packed_batch_mode_r8996b.load()==2?softmax_cast_run_r8996b:packed_run_submit_r8996b)(r,run,qf,packed_values,sink,output,batch_logits_r8996b,batch_probs_r8996b,batch_result_r8996b);
    batch_queries_r8996b+=run.count;++next_run_r8996b;t+=run.count-1;continue;
   }
   if(prefix_end_r9054[t]){if(prefix_stop_r9054)throw std::logic_error("r9054 overlapping prefix");prefix_begin_r9054=t;prefix_stop_r9054=prefix_end_r9054[t];}
   ++scalar_queries_r8996b;size_t n=selected[t].size();scalar_ops_r8996b+=n?4:2;auto logits=reuse?logits_storage->narrow({{0,0,heads*(n+1)}})->view({heads,n+1}):alloc({heads,n+1});logits->narrow({{1,n,1}})->copy_from(r,sink->view({heads,1}));
   rt::Tensor values;
   if(n){values=packed_values->narrow({{0,packed_offsets[t],n}});
    ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qf->narrow({{0,t,1}})->view({heads,512}),values->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1.f/std::sqrt(512.f)},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits->narrow({{1,0,n}}));
   }
   auto probabilities=reuse?probability_storage->narrow({{0,0,heads*(n+1)}})->view({heads,n+1}):alloc({heads,n+1});ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{1},std::optional<infini::ops::DataType>{},probabilities);
   auto result=prefix_stop_r9054?prefix_results_r9054->narrow({{0,t-prefix_begin_r9054,1}})->view({heads,512}):(reuse?result_storage:alloc({heads,512}));
   if(n)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),probabilities->narrow({{1,0,n}}),values,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},result);
   else{std::vector<float>zero(heads*512,0);r.memcpyH2D(result->data(),zero.data(),zero.size()*4,false);}
   if(prefix_stop_r9054){
    if(t+1==prefix_stop_r9054){size_t R=prefix_stop_r9054-prefix_begin_r9054;auto from=prefix_results_r9054->narrow({{0,0,R}});auto to=output->narrow({{0,prefix_begin_r9054,R}});
     ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),from,to);
     auto&c=scalar_cast_counts_r9054.at(r.device().index());++c[0];c[1].fetch_add(R);c[2].fetch_add(R-1);skipped_casts_r9054+=R-1;prefix_stop_r9054=0;}
   }else{
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),result,output->narrow({{0,t,1}})->view({heads,512}));
   }

  }if(prefix_stop_r9054)throw std::logic_error("r9054 unfinished prefix");scalar_ops_r8996b-=skipped_casts_r9054;r.syncStream();if(tokens>64){auto&c=reference_scratch_counts_r8954.at(r.device().index());++c[reuse?1:0];c[reuse?3:2].fetch_add(reuse?3:3*tokens);}if(tokens>64){auto&c=packed_batch_counts_r8996b.at(r.device().index());c[0].fetch_add(packed_batch_r8996b);c[1].fetch_add(packed_requested_r8996b&&!packed_batch_r8996b);c[2].fetch_add(scalar_queries_r8996b);c[3].fetch_add(next_run_r8996b);c[4].fetch_add(batch_queries_r8996b);c[5].fetch_add(scalar_ops_r8996b+(packed_batch_mode_r8996b.load()==2?2*batch_queries_r8996b+2*next_run_r8996b:4*next_run_r8996b));c[6].fetch_add(packed_batch_r8996b?3:0);}return output;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}

extern "C" int deep_reference_scratch_mode_r8953(int value){if(value!=0&&value!=1)return 1;infinilm::models::deepseek_v4::reference_scratch_mode_r8953.store(value);return 0;}

extern "C" int deep_reference_scratch_counts_r8954(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return 1;for(size_t k=0;k<4;++k)out[k]=infinilm::models::deepseek_v4::reference_scratch_counts_r8954[d][k].load();return 0;}

extern "C" int deep_reference_packed_batch_mode_r8996b(int v){if(v<0||v>2)return 1;infinilm::models::deepseek_v4::packed_batch_mode_r8996b.store(v);return 0;}
extern "C" int deep_reference_packed_batch_counts_r8996b(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=7)return 1;for(size_t k=0;k<7;++k)out[k]=infinilm::models::deepseek_v4::packed_batch_counts_r8996b[d][k].load();return 0;}

extern "C" int deep_sink_broadcast_mode_r9040(int v){if(v!=0&&v!=1)return 1;infinilm::models::deepseek_v4::sink_broadcast_mode_r9040.store(v);return 0;}
extern "C" int deep_sink_broadcast_counts_r9040(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t k=0;k<3;++k)out[k]=infinilm::models::deepseek_v4::sink_broadcast_counts_r9040[d][k].load();return 0;}

extern "C" int deep_scalar_cast_mode_r9054(int v){if(v!=0&&v!=1)return 1;infinilm::models::deepseek_v4::scalar_cast_mode_r9054.store(v);return 0;}
extern "C" int deep_scalar_cast_counts_r9054(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t k=0;k<3;++k)out[k]=infinilm::models::deepseek_v4::scalar_cast_counts_r9054[d][k].load();return 0;}

extern "C" int deep_values_reuse_mode_r9083(int v){if(v!=0&&v!=1)return 1;infinilm::models::deepseek_v4::values_reuse_mode_r9083.store(v);return 0;}
extern "C" int deep_values_reuse_counts_r9083(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=2)return 1;for(size_t k=0;k<2;++k)out[k]=infinilm::models::deepseek_v4::values_reuse_counts_r9083[d][k].load();return 0;}
