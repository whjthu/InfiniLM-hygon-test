import pathlib,json,hashlib,sys,importlib.util,numpy as np
r=pathlib.Path.cwd();p=r/'Deep-unique-SWA-private-r8983';src=r/'InfiniLM-hygon-test/code/bw1000/deep-reference-scratch/Deep_latent_maps_r8946.py';sp=importlib.util.spec_from_file_location('maps',src);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def plan(selected,lengths,rows):
 if not rows or rows>1048576 or len(lengths)!=len(selected):return None
 seen={};unique=[];offsets=[]
 for ids,ln in zip(selected,lengths):
  if ln[1]!=0:return None
  for j,x in enumerate(ids):
   if x<0 or x>=rows or(j and x!=ids[j-1]+1):return None
   if x not in seen:seen[x]=len(unique);unique.append(x)
 for ids in selected:
  off=seen[ids[0]]if ids else 0;offsets.append(off)
  if unique[off:off+len(ids)]!=ids:return None
 return unique,offsets
checks=[]
for kind in ['swa','hca128','csa4']:
 si,ci,lens,cr,_=m.maps(kind);selected=[a[:int(l[0])].tolist()+[int(v)+2048 for v in b[:int(l[1])]]for a,b,l in zip(si,ci,lens)];z=plan(selected,lens.tolist(),2048)
 if kind=='swa':
  assert z is not None;unique,offsets=z;assert len(unique)==2048 and sum(map(len,selected))==229632
  assert all(unique[a:a+len(v)]==v for a,v in zip(offsets,selected));assert unique==list(range(2048));checks.append({'kind':kind,'eligible':True,'originalrows':229632,'unique_rows':2048,'all2048_queries_originalorderedIDs_exact':True})
 else:assert z is None;checks.append({'kind':kind,'eligible':False,'reason':'compressedlengthnonzero/fallback'})
for ids in [[126,127,0,1],[1,3,4],[3,2,1],[2,2],[0,128]]:assert plan([ids],[[len(ids),0]],128)is None
assert plan([[124,125,126,127]],[[4,0]],128)==([124,125,126,127],[0])
assert plan([[],[0],[0,1]],[[0,0],[1,0],[2,0]],128)==([0,1],[0,0,0])
# Stable union preserves disjoint request physicalranges without sorting.
assert plan([[100,101],[100,101,102],[2,3]],[[2,0],[3,0],[2,0]],128)==([100,101,102,2,3],[0,0,3])
q={'CPU_only':True,'GPU':False,'native_build':False,'CXX_algorithm_not_yet_compiled':True,'shadow_plan_contract':checks,'negative_ringwrap_noncontiguous_reverse_duplicate_outofrange':'fallback','valid_past_window_bank_end':'eligibleexact','emptyquery':'eligible noGemm branch retained','stable_first_seen_union_no_sort':True,'all_original_perquery_keysets_and_order_preserved':True,'runtime_stride':'Existingr374 output contiguous FP32[materialrows,512]; everyquery values narrow remains[n,512] stride[512,1], originalGemm/Cast/logit/prob/result strides untouched. Inputbankactualstrides stillpassedunchangedtor374.','ring_scope':'Anyquery physicalIDwrap/nonascending/noncontiguous =>entirecall fallback. No ringrebasing, stale/future cache ID manufacture, or readingunselectedwholecache.','lifetime':'UniqueGPU ids andvalues allocatedthroughoriginalalloc/held untiloriginalr.syncStream; same Runtime.stream and original exceptiondrain. CPUvectors aliveuntildrain; keysimmutablethroughcall.','bindings':{'acceptedsource_SHA256':hashlib.sha256((r/'InfiniLM-hygon-test/code/bw1000/deep-reference-scratch/deep_latent_reference_scratch_r8954.cpp').read_bytes()).hexdigest(),'candidate_SHA256':hashlib.sha256((p/'deep_latent_unique_SWA_r8983.cpp').read_bytes()).hexdigest(),'runner_SHA256':hashlib.sha256((p/'Deep_unique_SWA_component_r8983.py').read_bytes()).hexdigest(),'fixturemaps_SHA256':hashlib.sha256(src.read_bytes()).hexdigest()},'remaining_before_build':'ActualCPU219MD confirms r356header51f501 and r374gather.o e101ffa unique symbol; freshMD required newhostbuild, r374historicalsource mapping UNKNOWN; noSDK/kernelrebuild.','GPU_plan':'Samecurrent923scratchON/8933OFF, oneT2048SWAfixture/8ownerseeds/2warm+8timedABBA/BAAB+restoreA, existing originalGemmSoftmaxCast, fullattentionenclosingwall/finite/ownrepeat/crossarmerror/normalpost; no additional explicit canary in inherited runner. CPUreceipt/native/manifest unavailable so NOT RUNNABLE/NOT GPU GO.'}
(p/'Deep-unique-SWA-source-contract-r8983.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
