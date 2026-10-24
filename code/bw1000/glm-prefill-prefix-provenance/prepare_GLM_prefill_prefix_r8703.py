import pathlib,hashlib,json
r=pathlib.Path('.');producer=r/'glm_shared_validation_cache_r5814.cpp';dense=r/'glm_moe_dsa_dense_modes_r48.cpp'
assert hashlib.sha256(producer.read_bytes()).hexdigest()=='4d7f4a9293cff492c1e8835560d345ea7aeace58d11b2b303188b312a0934962'
assert hashlib.sha256(dense.read_bytes()).hexdigest()=='5d27978cced7d942a0f91ee8233aae30a29198cc8f494779db788e3f88f4a435'
p=producer.read_text();s=dense.read_text()
p='#include "glm_prefill_prefix_proof_r8703.hpp"\n'+p
needle='else{auto mapped_ids=work.ids(mapped)->view(chosen->shape());result=sparse_attention_mapped_bounded(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16);}'
replacement='''else{auto mapped_ids=work.ids(mapped)->view(chosen->shape());
    const char*plan_flag8703=std::getenv("GLM_PREFILL_PREFIX_PLAN_R8703");
    if(plan_flag8703&&std::string(plan_flag8703)!="0"&&std::string(plan_flag8703)!="1")throw std::invalid_argument("8703 prefix-plan flag 0/1");
    if(plan_flag8703&&std::string(plan_flag8703)=="1"&&complete_prefix&&capacity_<=2048&&prefix<=2048){
     PrefillPrefixProofR8703 proof(r,mapped_ids,ids,request.row,request.past,count,capacity_,logical,mapped);
     if(proof.complete())result=sparse_attention_mapped_bounded_proven_r8703(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16,proof,request.row,capacity_);
     else result=sparse_attention_mapped_bounded(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16);
    }else result=sparse_attention_mapped_bounded(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16);
   }'''
assert p.count(needle)==1;p=p.replace(needle,replacement)
s='#include "glm_prefill_prefix_proof_r8703.hpp"\n#include <array>\n#include <atomic>\n'+s
s=s.replace('thread_local int dense_mode_r48=7;', 'std::array<std::atomic<uint64_t>,8>proven_rows8703{},proven_tokens8703{},skipped_readback_bytes8703{};\nthread_local int dense_mode_r48=7;')
sig='size_t past,float scale){\n r.assertOwnerThread();'
assert s.count(sig)==1;s=s.replace(sig,'size_t past,float scale,const PrefillPrefixProofR8703*proof8703=nullptr,size_t row8703=0,size_t capacity8703=0){\n r.assertOwnerThread();',1)
a=s.index(' size_t tokens=q->size(0),heads=q->size(1),width=indices->size(1);');b=s.index(' std::vector<rt::Tensor> held;',a)
old=s[a:b]
new=''' size_t tokens=q->size(0),heads=q->size(1),width=indices->size(1);
 const bool trusted8703=proof8703&&dense_mode_r48==2;
 if(trusted8703&&!proof8703->matches(r,indices,positions,row8703,past,tokens,capacity8703,k->size(0)))throw std::invalid_argument("8703 prefix-plan owner/row/Tensor binding differs");
 std::vector<int64_t>logical;
 std::vector<std::vector<int64_t>>selected;
 if(!trusted8703){
'''+old.replace(' size_t tokens=q->size(0),heads=q->size(1),width=indices->size(1);',' ').replace('std::vector<int64_t>logical(k->size(0));','logical.resize(k->size(0));').replace('std::vector<std::vector<int64_t>> selected(tokens);','selected.resize(tokens);')+''' }
'''
s=s[:a]+new+s[b:]
needle='bool dense=tokens>1 && k->size(0)==past+tokens && k->size(0)<=2048;'
assert s.count(needle)==1;s=s.replace(needle,'bool dense=trusted8703||(tokens>1 && k->size(0)==past+tokens && k->size(0)<=2048);')
s=s.replace('for(size_t t=0;dense&&t<tokens;++t){dense=selected[t].size()==past+t+1;','for(size_t t=0;dense&&!trusted8703&&t<tokens;++t){dense=selected[t].size()==past+t+1;')
needle='\n}\n\nnamespace infinilm::models::glm_moe_dsa {\nextern "C" int modern_glm_dense_mode_r48_set'
assert s.count(needle)==1
entry='''
rt::Tensor sparse_attention_mapped_bounded_proven_r8703(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&indices,const rt::Tensor&positions,size_t past,float scale,const PrefillPrefixProofR8703&proof,size_t row,size_t capacity){
 if(!positions)throw std::invalid_argument("8703 mapped key positions missing");
 // Preserve non-mode2 paths, including original mode8 selected-vector use.
 if(dense_mode_r48!=2)return sparse_impl(r,q,k,v,indices,positions,past,scale);
 auto result=sparse_impl(r,q,k,v,indices,positions,past,scale,&proof,row,capacity);
 const int device=r.device().index();if(device>=0&&device<8){proven_rows8703[size_t(device)].fetch_add(1,std::memory_order_relaxed);proven_tokens8703[size_t(device)].fetch_add(q->size(0),std::memory_order_relaxed);skipped_readback_bytes8703[size_t(device)].fetch_add((indices->numel()+positions->numel())*8,std::memory_order_relaxed);}return result;
}
'''
s=s.replace(needle,entry+needle)
s+='\nextern \"C\" uint64_t modern_glm_prefill_prefix_plan_counts_r8703(int device,int kind){using namespace infinilm::models::glm_moe_dsa;if(device<0||device>=8||kind<0||kind>2)return 0;return (kind==0?proven_rows8703[size_t(device)]:kind==1?proven_tokens8703[size_t(device)]:skipped_readback_bytes8703[size_t(device)]).load(std::memory_order_relaxed); }\n'
(r/'glm_prefill_prefix_cache_r8703.cpp').write_text(p,encoding='utf-8',newline='\n');(r/'glm_prefill_prefix_dense_r8703.cpp').write_text(s,encoding='utf-8',newline='\n')
manifest={'CPU_only':True,'GPU_GO':False,'compiled':False,'parent_provider_SHA256':'1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d','parent_source_SHA256':{producer.name:hashlib.sha256(producer.read_bytes()).hexdigest(),dense.name:hashlib.sha256(dense.read_bytes()).hexdigest()},'sources_SHA256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['glm_prefill_prefix_proof_r8703.hpp','glm_prefill_prefix_cache_r8703.cpp','glm_prefill_prefix_dense_r8703.cpp']},'scope':'Two existing TUs plus private proof header. DefaultOFF. CPU actual logical/mapped plan must certify every query complete ascending causal prefix. Same Runtime/row/Tensor identity/read-only lifetime pattern as PrefixProof1645; no new math/sync/stream/kernel/SDK/cache. Noncomplete and nonmode2 stay old path. Benefit UNKNOWN; native compile and actual scope hit not yet qualified.'}
(r/'GLM-prefill-prefix-CPU-source-r8703.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
