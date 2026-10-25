from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent
state='''
namespace { struct AttentionCohort9054 {int enabled=0;uint64_t counts[4]{};};thread_local std::array<AttentionCohort9054,8> attention_cohort9054; }
extern "C" int glm_attention_cohort_policy_r9054(int d,int mode){if(d<0||d>=8||mode<0||mode>1)return -1;attention_cohort9054[d].enabled=mode;return 0;}
extern "C" int glm_attention_cohort_counts_r9054(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return -1;for(size_t j=0;j<4;++j)out[j]=attention_cohort9054[d].counts[j];return 0;}
extern "C" int glm_attention_cohort_choose_r9054(int d,int entry,int eligible){if(d<0||d>=8||entry<0||entry>1)return 0;auto&s=attention_cohort9054[d];int selected=eligible&&s.enabled;++s.counts[entry*2+selected];return selected;}
'''
def body(entry):
 first='const size_t cohortP=keys[0]->size(0);bool equalP=true;for(const auto&key:keys)equalP=equalP&&key->size(0)==cohortP;const bool cohortEligible=backend==1&&equalP&&(B==8||B==16);'if entry==0 else'const size_t cohortP=P;const bool cohortEligible=B==8||B==16;'
 kv='auto kr=keys[i],vr=values[i];'if entry==0 else'auto kr=k->narrow({{0,i,1}})->view({cohortP,8,256}),vr=v->narrow({{0,i,1}})->view({cohortP,8,256});'
 return '''
 // Default OFF. Only equal-P cohorts; original per-row QK/PV shapes/providers.
 '''+first+'''
 if(glm_attention_cohort_choose_r9054(r.device().index(),'''+str(entry)+''',cohortEligible)) {
  auto cdots=alloc({B*8,1,cohortP},rt::DataType::kBFloat16),clogits=alloc({B*8,1,cohortP},rt::DataType::kFloat32),cprob=alloc({B*8,1,cohortP},rt::DataType::kFloat32),cpb=alloc({B*8,1,cohortP},rt::DataType::kBFloat16),cresults=alloc({B,8,256},rt::DataType::kBFloat16);
  std::vector<rt::Tensor> qrows,krows,vrows,drows,prows,rrows;
  for(size_t i=0;i<B;++i) {
   auto qr=q->narrow({{0,i,1}});'''+kv+'''
   qrows.push_back(qr->view({8,1,256}));krows.push_back(kr->permute({1,2,0}));vrows.push_back(vr->permute({1,0,2}));drows.push_back(cdots->narrow({{0,i*8,8}}));prows.push_back(cpb->narrow({{0,i*8,8}}));rrows.push_back(cresults->narrow({{0,i,1}})->view({8,1,256}));
   work.held.insert(work.held.end(),{qrows.back(),krows.back(),vrows.back(),drows.back(),prows.back(),rrows.back()});
   if(drows.back()->shape()!=rt::Shape({8,1,cohortP})||prows.back()->shape()!=rt::Shape({8,1,cohortP})||!drows.back()->is_contiguous()||!prows.back()->is_contiguous()||reinterpret_cast<uintptr_t>(drows.back()->data())%16||reinterpret_cast<uintptr_t>(prows.back()->data())%16)throw std::logic_error("9054 cohort view layout");
  }
  for(size_t i=0;i<B;++i)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),qrows[i],krows[i],std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},drows[i]);
  if(glm_scale_round_r392(cdots->data(),clogits->data(),nullptr,B*8*cohortP,1.f/16,r.stream()))throw std::runtime_error("9054 original scale round");
  ex::submit<infini::ops::Softmax>(r,infini::ops::Softmax::DefaultConfig(r.device().type()),clogits,int64_t{2},std::optional<infini::ops::DataType>{},cprob);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),cprob,cpb);
  for(size_t i=0;i<B;++i)ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),prows[i],vrows[i],std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},rrows[i]);
  out->copy_from(r,cresults);
  r.syncStream();'''+('++batch_stats5096.at(r.device().index())[0];batch_stats5096.at(r.device().index())[1].fetch_add(B);'if entry==1 else'')+'''return out;
 }
'''
for entry,old in enumerate(['glm_reduced_output_5555_r9032.cpp','glm_reduced_output_5096_r9032.cpp']):
 s=(r/old).read_text();s=s.replace('namespace infinilm::models::glm_moe_dsa {', (state if entry==0 else'extern "C" int glm_attention_cohort_choose_r9054(int,int,int);\n')+'namespace infinilm::models::glm_moe_dsa {',1)
 marker=' for(size_t i=0;i<B;++i){'if entry==0 else'for(size_t i=0;i<B;++i){';assert s.count(marker)==1
 s=s.replace(marker,body(entry)+marker,1)
 new='glm_attention_cohort_'+('5555'if entry==0 else'5096')+'_r9054.cpp';(r/new).write_bytes(s.encode())
print(json.dumps({'sources':['glm_attention_cohort_5555_r9054.cpp','glm_attention_cohort_5096_r9054.cpp'],'GPU_TUs':0,'HOST_TUs':2,'mechanism':'samePscaleSoftmaxCast rowsbatch;originalQK/PVunchanged;heldtooriginaldrain;defaultOFF'}))
