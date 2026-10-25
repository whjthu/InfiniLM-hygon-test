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
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::atomic<int> reference_scratch_mode_r8953{0};std::array<std::array<std::atomic<uint64_t>,4>,8> reference_scratch_counts_r8954{};}

namespace {
std::atomic<int> unique_swa_mode_r8983{0};
std::atomic<int> grouped_swa_mode_r8984{0};
std::array<std::array<std::atomic<uint64_t>,7>,8> grouped_swa_counts_r8984{};
std::array<std::array<std::atomic<uint64_t>,4>,8> unique_swa_counts_r8983{};
// Stable first-use bank: NEVER sort, alter, or deduplicate a query's keys.
// Wrapped ring IDs, compressed keys, noncontiguous physical ranges, or
// noncontiguous compact-bank views all take the unchanged original path.
bool unique_swa_plan_r8983(const std::vector<std::vector<int64_t>>& selected,
 const std::vector<int32_t>& lengths,size_t swa_rows,
 std::vector<int64_t>& unique_ids,std::vector<size_t>& offsets){
 if(!swa_rows||swa_rows>1048576||lengths.size()!=selected.size()*2)return false;
 const size_t missing=std::numeric_limits<size_t>::max();
 std::vector<size_t> seen(swa_rows,missing);offsets.assign(selected.size(),0);
 for(size_t t=0;t<selected.size();++t){
  if(lengths[2*t+1]!=0)return false;
  const auto& ids=selected[t];
  for(size_t j=0;j<ids.size();++j){
   const int64_t id=ids[j];
   if(id<0||size_t(id)>=swa_rows||(j&&id!=ids[j-1]+1))return false;
   if(seen[size_t(id)]==missing){seen[size_t(id)]=unique_ids.size();unique_ids.push_back(id);}
  }
 }
 for(size_t t=0;t<selected.size();++t){
  const auto& ids=selected[t];if(ids.empty())continue;
  offsets[t]=seen[size_t(ids[0])];
  for(size_t j=0;j<ids.size();++j)
   if(seen[size_t(ids[j])]!=offsets[t]+j||unique_ids[offsets[t]+j]!=ids[j])return false;
 }
 return true;
}
struct GroupedSwaRunR8984 { size_t query_begin, count, bank_begin; };
// Deliberately narrow to the actual B4/I512 map. Reject compressed/ring/
// different row layouts for the WHOLE call; never infer a request from values.
static bool grouped_swa_runs_r8984(
 const std::vector<std::vector<int64_t>>& selected,
 const std::vector<int32_t>& lengths, size_t bank_rows,
 const std::vector<size_t>& unique_offsets,
 std::vector<GroupedSwaRunR8984>& runs) {
 if(selected.size()!=2048 || lengths.size()!=4096 || bank_rows!=2048 ||
    unique_offsets.size()!=2048)return false;
 for(size_t t=0;t<2048;++t){
  const size_t request=t/512,pos=t%512,n=std::min(pos+1,size_t{128});
  if(lengths[2*t]!=int32_t(n)||lengths[2*t+1]!=0||selected[t].size()!=n)return false;
  for(size_t j=0;j<n;++j)
   if(selected[t][j]!=int64_t(request*512+pos+1-n+j))return false;
 }
 for(size_t request=0;request<4;++request){
  const size_t first=request*512+128,offset=unique_offsets[first];
  for(size_t j=0;j<384;++j)if(unique_offsets[first+j]!=offset+j)return false;
  runs.push_back({first,384,offset});
 }
 return true;
}
// Prerequisite unique_values: original BF16->FP32 r374 gather, shape[2048,512].
// This function submits exactly two original SDK Gemm + Softmax + Cast per run.
// alloc below MUST be the parent's held allocator; scratch may be allocated
// once for max_count=384 and reused sequentially across four runs.
template<class HeldAlloc>
static void grouped_swa_run_r8984(rt::Runtime&r,const GroupedSwaRunR8984&run,
 const rt::Tensor&qf,const rt::Tensor&unique_values,const rt::Tensor&sink,
 const rt::Tensor&output,HeldAlloc&alloc,const rt::Tensor&logits_store,
 const rt::Tensor&prob_store,const rt::Tensor&result_store) {
 const size_t R=run.count,H=qf->size(1),N=128,D=512;
 // Prove span using original physical backing. as_strided has NO safety check.
 if(H!=8||R!=384||unique_values->shape()!=rt::Shape{2048,512}||
    !unique_values->is_contiguous()||run.bank_begin+R+N-1>2048)
  throw std::logic_error("grouped SWA span differs");
 auto bank=unique_values->narrow({{0,run.bank_begin,R+N-1}});
 auto values=bank->as_strided({R,N,D},{512,512,1}); // readonly overlap
 auto qr=qf->narrow({{0,run.query_begin,R}}); // [R,H,D], originalF32
 auto logits=logits_store->narrow({{0,0,R}}); // [R,H,129], contiguous
 auto probs=prob_store->narrow({{0,0,R}});
 auto result=result_store->narrow({{0,0,R}}); // [R,H,512]
 // Preserve the original perquery sink copy path; no new broadcast or sink cache.
 for(size_t t=0;t<R;++t)
  logits->narrow({{0,t,1}})->view({H,N+1})->narrow({{1,N,1}})->copy_from(r,sink->view({H,1}));
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
  qr,values->permute({0,2,1}),std::optional<rt::Tensor>{},
  std::optional<float>{1.f/std::sqrt(512.f)},std::optional<float>{0},
  std::optional<int>{},std::optional<int>{},logits->narrow({{2,0,N}}));
 ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),
  logits,int64_t{2},std::optional<infini::ops::DataType>{},probs);
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),
  probs->narrow({{2,0,N}}),values,std::optional<rt::Tensor>{},
  std::optional<float>{1},std::optional<float>{0},std::optional<int>{},
  std::optional<int>{},result);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),
  result,output->narrow({{0,run.query_begin,R}}));
 // NO per-run sync. Parent's final/exception drain retains all storage.
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
 std::vector<int64_t> unique_ids_r8983;std::vector<size_t> unique_offsets_r8983;
 const bool grouped_requested_r8984=grouped_swa_mode_r8984.load()==1;
 const bool unique_requested_r8983=(unique_swa_mode_r8983.load()==1||grouped_requested_r8984)&&reference_scratch_mode_r8953.load()==1&&tokens>64;
 bool unique_r8983=unique_requested_r8983&&!packed_ids.empty()&&unique_swa_plan_r8983(selected,hl,swa->size(0),unique_ids_r8983,unique_offsets_r8983);
 std::vector<GroupedSwaRunR8984> grouped_runs_r8984;
 const bool grouped_r8984=grouped_requested_r8984&&unique_r8983&&heads==8&&sw==128&&cw==1&&compressed->size(0)==1&&grouped_swa_runs_r8984(selected,hl,swa->size(0),unique_offsets_r8983,grouped_runs_r8984);
 // An unsupported grouped request takes the untouched original packed path.
 if(grouped_requested_r8984&&!grouped_r8984)unique_r8983=false;
 const auto& material_ids_r8983=unique_r8983?unique_ids_r8983:packed_ids;
 const auto& material_offsets_r8983=unique_r8983?unique_offsets_r8983:packed_offsets;

 if(packed_ids.size()>std::numeric_limits<size_t>::max()/2048)throw std::invalid_argument("DeepSeek packed latent selection size overflows");
 std::vector<rt::Tensor>held;auto alloc=[&](rt::Shape shape,rt::DataType dt=rt::DataType::kFloat32){auto x=rt::Tensor::empty(r,shape,dt);held.push_back(x);return x;};
 try{auto output=alloc({tokens,heads,512},q->dtype());
  auto qf=alloc({tokens,heads,512});ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),q,qf);
  rt::Tensor packed_values;
  if(!material_ids_r8983.empty()){
   auto ids=alloc({material_ids_r8983.size()},rt::DataType::kInt64);r.memcpyH2D(ids->data(),material_ids_r8983.data(),material_ids_r8983.size()*8,true);
   packed_values=alloc({material_ids_r8983.size(),512});
   int status=deep_packed_gather_r374(packed_values->data(),swa->data(),compressed->data(),ids->data(),material_ids_r8983.size(),swa->size(0),swa->stride(0),swa->stride(1),compressed->stride(0),compressed->stride(1),r.stream());
   if(status)throw std::runtime_error("DeepSeek fused packed gather failed: "+std::to_string(status));
  }
  // Proposal only: same-stream scratch, retained in held until original completion drain.
  const bool reuse=reference_scratch_mode_r8953.load()==1&&tokens>64;
  rt::Tensor logits_storage,probability_storage,result_storage;
  if(reuse){size_t max_n=0;for(const auto&v:selected)if(v.size()>max_n)max_n=v.size();logits_storage=alloc({heads*(max_n+1)});probability_storage=alloc({heads*(max_n+1)});result_storage=alloc({heads,512});}
  rt::Tensor grouped_logits_r8984,grouped_probs_r8984,grouped_result_r8984;
  if(grouped_r8984){grouped_logits_r8984=alloc({384,8,129});grouped_probs_r8984=alloc({384,8,129});grouped_result_r8984=alloc({384,8,512});}
  size_t next_run_r8984=0,scalar_queries_r8984=0,batch_queries_r8984=0,scalar_ops_r8984=0;
  for(size_t t=0;t<tokens;++t){
   if(grouped_r8984&&next_run_r8984<grouped_runs_r8984.size()&&t==grouped_runs_r8984[next_run_r8984].query_begin){
    const auto&run=grouped_runs_r8984[next_run_r8984];
    grouped_swa_run_r8984(r,run,qf,packed_values,sink,output,alloc,grouped_logits_r8984,grouped_probs_r8984,grouped_result_r8984);
    batch_queries_r8984+=run.count;++next_run_r8984;t+=run.count-1;continue;
   }
   ++scalar_queries_r8984;
   size_t n=selected[t].size();scalar_ops_r8984+=n?4:2;auto logits=reuse?logits_storage->narrow({{0,0,heads*(n+1)}})->view({heads,n+1}):alloc({heads,n+1});logits->narrow({{1,n,1}})->copy_from(r,sink->view({heads,1}));
   rt::Tensor values;
   if(n){values=packed_values->narrow({{0,material_offsets_r8983[t],n}});
    ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qf->narrow({{0,t,1}})->view({heads,512}),values->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1.f/std::sqrt(512.f)},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits->narrow({{1,0,n}}));
   }
   auto probabilities=reuse?probability_storage->narrow({{0,0,heads*(n+1)}})->view({heads,n+1}):alloc({heads,n+1});ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),logits,int64_t{1},std::optional<infini::ops::DataType>{},probabilities);
   auto result=reuse?result_storage:alloc({heads,512});
   if(n)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),probabilities->narrow({{1,0,n}}),values,std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},result);
   else{std::vector<float>zero(heads*512,0);r.memcpyH2D(result->data(),zero.data(),zero.size()*4,false);}
   ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),result,output->narrow({{0,t,1}})->view({heads,512}));
  }r.syncStream();if(tokens>64){auto&c=reference_scratch_counts_r8954.at(r.device().index());++c[reuse?1:0];c[reuse?3:2].fetch_add(reuse?3:3*tokens);}if(tokens>64){auto&uc=unique_swa_counts_r8983.at(r.device().index());uc[0].fetch_add(unique_r8983);uc[1].fetch_add(unique_requested_r8983&&!unique_r8983);uc[2].fetch_add(packed_ids.size());uc[3].fetch_add(material_ids_r8983.size());}if(tokens>64){auto&gc=grouped_swa_counts_r8984.at(r.device().index());gc[0].fetch_add(grouped_r8984);gc[1].fetch_add(grouped_requested_r8984&&!grouped_r8984);gc[2].fetch_add(scalar_queries_r8984);gc[3].fetch_add(next_run_r8984);gc[4].fetch_add(batch_queries_r8984);gc[5].fetch_add(scalar_ops_r8984+next_run_r8984*4);gc[6].fetch_add(grouped_r8984?3:0);}return output;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}

extern "C" int deep_reference_scratch_mode_r8953(int value){if(value!=0&&value!=1)return 1;infinilm::models::deepseek_v4::reference_scratch_mode_r8953.store(value);return 0;}

extern "C" int deep_reference_scratch_counts_r8954(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return 1;for(size_t k=0;k<4;++k)out[k]=infinilm::models::deepseek_v4::reference_scratch_counts_r8954[d][k].load();return 0;}

extern "C" int deep_reference_unique_swa_mode_r8983(int v){if(v!=0&&v!=1)return 1;infinilm::models::deepseek_v4::unique_swa_mode_r8983.store(v);return 0;}
extern "C" int deep_reference_unique_swa_counts_r8983(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return 1;for(size_t k=0;k<4;++k)out[k]=infinilm::models::deepseek_v4::unique_swa_counts_r8983[d][k].load();return 0;}

extern "C" int deep_reference_grouped_swa_mode_r8984(int v){if(v!=0&&v!=1)return 1;infinilm::models::deepseek_v4::grouped_swa_mode_r8984.store(v);return 0;}
extern "C" int deep_reference_grouped_swa_counts_r8984(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=7)return 1;for(size_t k=0;k<7;++k)out[k]=infinilm::models::deepseek_v4::grouped_swa_counts_r8984[d][k].load();return 0;}
