from pathlib import Path
import json,hashlib,ast
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cpu=json.loads((r/'GLM-attention-cohort-CPU-observed-r9054.json').read_bytes());proof=cpu['receipt'];assert proof['completed'];ds=proof['library_SHA256']
def rename(t):return t.replace('lookahead','attention_cohort').replace('r9044','r9054').replace('glm-gate-attention_cohort','glm-attention-cohort')
files=[]
for old in ['GLM_lookahead_layer_r9044.py','GLM_lookahead_layer_r9044_inside.sh','controller_GLM_lookahead_layer_r9044.py','launch_GLM_lookahead_layer_r9044.py']:
 new=rename(old);s=rename((r/old).read_text())
 if old=='GLM_lookahead_layer_r9044.py':
  s=s.replace("DSO_SHA='c1ccce0fddd3967fcda441b5db93068ed689a39c49885122124eea2498eb0077'",'DSO_SHA='+repr(ds))
  s=s.replace('glm_gate_attention_cohort_policy_r9042','glm_attention_cohort_policy_r9054').replace('glm_gate_attention_cohort_counts_r9042','glm_attention_cohort_counts_r9054')
  s=s.replace('lib.glm_attention_cohort_policy_r9054.argtypes=[I]','lib.glm_attention_cohort_policy_r9054.argtypes=[I,I]').replace('lib.glm_attention_cohort_counts_r9054.argtypes=[ctypes.POINTER(ctypes.c_uint64),Z]','lib.glm_attention_cohort_counts_r9054.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]')
  s=s.replace('(ctypes.c_uint64*2)();ck(lib.glm_attention_cohort_counts_r9054(v,2));return [v[0],v[1],0]','(ctypes.c_uint64*4)();ck(lib.glm_attention_cohort_counts_r9054(rank,v,4));return [v[0]+v[2],v[1]+v[3],0]')
  for mode in['99','0','mode']:s=s.replace('glm_attention_cohort_policy_r9054('+mode+')','glm_attention_cohort_policy_r9054(rank,'+mode+')')
  s=s.replace("proof['GPU_TUs_compiled_on_CPU']==1 and proof['HOST_TUs_compiled']==0 and len(proof['reused_objects_SHA256'])==27","proof['GPU_TUs_compiled_on_CPU']==0 and proof['HOST_TUs_compiled']==2 and len(proof['reused_objects_SHA256'])==26")
  s=s.replace('Gate one-block attention_cohort only; originalgate/down ISA bound; directPV selector alwaysOFF; QK/attention unchanged; finaltokens not qualified','Attention sameP cohortscale/Softmax/Cast only; QK/PV eachroworiginalshape/dtype/provider/defaultconfig; originalscale392BF16round; originalMoE; directPVfixedOFF; notModelqualified')
 (r/new).write_bytes(s.encode());files.append(new)
for old in ['GLM-lookahead-layer-manifest-r9044.json','GLM-lookahead-layer-root-GO-r9044.json','GLM-lookahead-layer-execution-authorized-r9044.json']:
 d=json.loads((r/old).read_bytes());new=rename(old)
 if 'sources_SHA256'in d:
  d['sources_SHA256']={n:sha(r/n)for n in files};d.update(scope='Currentlayer38B8past512 originalattention vssamePscaleSoftmaxCast cohort;2HOSTonly;originalQK/PV/GPU/SDK/drains;11calls88cases4pairs',candidate_DSO_SHA256=ds,CPU_handle=proof['producer'],CPU_receipt_SHA256=cpu['receipt_SHA256']);d['DSO_SHA256'].pop('candidates/glm-gate-lookahead-r9044/libglm_gate_lookahead_r9044.so');d['DSO_SHA256']['candidates/glm-attention-cohort-r9054/libglm_attention_cohort_r9054.so']=ds;d['matrix']['candidate']='samePscaleSoftmaxCastcohort defaultOFF;QK/PVoriginal;directPVfixedOFF'
 elif 'DSO_SHA256'in d:d['DSO_SHA256']=ds
 (r/new).write_bytes((json.dumps(d,indent=2)+'\n').encode());files.append(new)
old='observe_GLM_lookahead_layer_r9044.py';(r/rename(old)).write_bytes(rename((r/old).read_text()).encode())
for n in files:
 if n.endswith('.py'):ast.parse((r/n).read_bytes())
 assert b'\r'not in(r/n).read_bytes()
print(json.dumps({'files':files,'DSO_SHA256':ds,'CPU_handle':proof['producer']}))
