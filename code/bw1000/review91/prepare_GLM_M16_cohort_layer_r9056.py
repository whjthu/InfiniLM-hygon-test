from pathlib import Path
import json,hashlib,ast
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cpu=json.loads((r/'GLM-M16-bridge-CPU-observed-r9056b.json').read_bytes());proof=cpu['receipt'];assert proof['completed'];ds=proof['library_SHA256']
def rename(t):return t.replace('GLM-attention_cohort-layer','GLM-M16-cohort-layer').replace('GLM_attention_cohort_layer','GLM_M16_cohort_layer').replace('r9054','r9056').replace('glm_attention_cohort_policy_r9056','glm_attention_cohort_policy_r9054').replace('glm_attention_cohort_counts_r9056','glm_attention_cohort_counts_r9054')
files=[]
for old in ['GLM_attention_cohort_layer_r9054.py','GLM_attention_cohort_layer_r9054_inside.sh','controller_GLM_attention_cohort_layer_r9054.py','launch_GLM_attention_cohort_layer_r9054.py']:
 new=rename(old);s=rename((r/old).read_text())
 if old=='GLM_attention_cohort_layer_r9054.py':
  s=s.replace("base=r/'candidates/glm-attention-cohort-r9056'","base=r/'candidates/glm-M16-cohort-r9056b'").replace("DSO_SHA='45a3ad06b789152c532abcaa27a45db582bf55669ec4f7a56e6ba08b9de3003a'",'DSO_SHA='+repr(ds))
  s=s.replace("proof['HOST_TUs_compiled']==2 and len(proof['reused_objects_SHA256'])==26","proof['HOST_TUs_compiled']==1 and len(proof['reused_objects_SHA256'])==27")
  s=s.replace(' lib.modern_glm_vector_policy_r8540.argtypes=[I,I]',' lib.modern_glm_vector_policy_r8540.argtypes=[I,I]\n lib.modern_glm_vector_counts_r8540.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]\n lib.glm_grouped_counts_r5123.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]')
  s=s.replace('ck(lib.glm_reduced_output_counts_r9032(rank,v,4));return [v[0],v[1],0,v[2],v[3],0]','ck(lib.glm_attention_cohort_counts_r9054(rank,v,4));return [v[0],v[1],0,v[2],v[3],0]')
  s=s.replace(' def agree(value):',' def moe_stats():\n  a=(ctypes.c_uint64*8)();b=(ctypes.c_uint64*5)();ck(lib.modern_glm_vector_counts_r8540(rank,a,8));ck(lib.glm_grouped_counts_r5123(rank,b,5));return list(a),list(b)\n def agree(value):')
  for a,b in [('((4096,6144)','((8192,6144)'),('((8,6144)','((16,6144)'),('.repeat(8).cuda()','.repeat(16).cuda()'),('full((8,),512','full((16,),512'),('8*6144+128','16*6144+128'),('view(8,6144)','view(16,6144)')]:assert a in s,a;s=s.replace(a,b)
  a='desc=(Z*24)(*[v for row in range(8)for v in [row,0,512]]);decode_desc=(Z*24)(*[v for row in range(8)for v in [row,512,1]])';assert a in s
  s=s.replace(a,'prefill_desc=[(Z*24)(*[v for row in range(start,start+8)for v in [row,0,512]])for start in [0,8]];decode_desc=(Z*48)(*[v for row in range(16)for v in [row,512,1]])')
  a='ck(lib.glm_layer_forward_r5670(ctx,4096,8,desc,V(x.data_ptr()),V(pos.data_ptr()),V(out.data_ptr()),0))';assert a in s
  s=s.replace(a,'\n   for cohort in range(2):ck(lib.glm_layer_forward_r5670(ctx,4096,8,prefill_desc[cohort],V(x[cohort*4096].data_ptr()),V(pos[cohort*4096].data_ptr()),V(out[cohort*4096].data_ptr()),0))')
  s=s.replace('ctx,8,8,decode_desc','ctx,16,16,decode_desc').replace("report['N']==8 and report['B']==8","report['N']==16 and report['B']==16")
  s=s.replace('before=stats();entry_before=entry_stats();torch.cuda.synchronize()','before=stats();entry_before=entry_stats();vector_before,grouped_before=moe_stats();torch.cuda.synchronize()')
  s=s.replace('actual=decode_out.cpu();after=stats();','actual=decode_out.cpu();vector_after,grouped_after=moe_stats();vector_delta=[a-b for a,b in zip(vector_after,vector_before)];grouped_delta=[a-b for a,b in zip(grouped_after,grouped_before)];assert vector_delta==[0,1,0,1,0,0,0,0]and grouped_delta==[2,0,0,2,0],("actual_R128_original_fallback",vector_delta,grouped_delta);after=stats();')
  s=s.replace('entry_delta[3:]==[1,0,0]','entry_delta[3:]==([0,1,0]if mode else[1,0,0])')
  s=s.replace("'wholelayer_native_wall_ms':report['wall_ms']","'actual_R128_vector_delta':vector_delta,'actual_R128_grouped_delta':grouped_delta,'wholelayer_native_wall_ms':report['wall_ms']")
  s=s.replace("'fixture_provenance']={", "'fixture_provenance']={'prefix_cohort_row_IDs':[list(range(8)),list(range(8,16))],'same_context_without_reset_between_prefix_cohorts':True,'logical_request_checks_per_TP_replica':176,'decode_rows':16,'routed_slots':128,'MoE_provider':'vector8540 fallback to f359 originalrouted19; notR64vector orMMAC',")
  s=s.replace('prefix[4096,6144] + separate M8 hidden[8,6144]','prefix[8192,6144] in2stable-rowB8cohorts + separate16independent hidden[16,6144]')
  s=s.replace("'only_one_weight_load':True", "'only_one_weight_load':True")
 elif old.startswith('controller'):
  s=s.replace("a['entry_delta']==[0,0,0,1,0,0]","a['entry_delta']==([0,0,0,0,1,0]if a['mode']else[0,0,0,1,0,0])and a['actual_R128_vector_delta']==[0,1,0,1,0,0,0,0]and a['actual_R128_grouped_delta']==[2,0,0,2,0]")
 (r/new).write_bytes(s.encode());files.append(new)
for old in ['GLM-attention_cohort-layer-manifest-r9054.json','GLM-attention_cohort-layer-root-GO-r9054.json','GLM-attention_cohort-layer-execution-authorized-r9054.json']:
 d=json.loads((r/old).read_bytes());new=rename(old)
 if 'sources_SHA256'in d:
  d['sources_SHA256']={n:sha(r/n)for n in files};d.update(scope='ROOT M16 target newgeometry:1HOSTbridgeB16,N4096 unchanged;twoB8prefixcallsstableIDs0..15;samePattentioncohort;R128originalgroupedfallback;176requestchecks',candidate_DSO_SHA256=ds,CPU_handle=proof['producer'],CPU_receipt_SHA256=cpu['receipt_SHA256'],root_extended_GO='Direct root NEW_TASK permits necessary1HOSTbridge afteractualT16cache/leasecapacityclosure;639PARK remainsimmutable');d['DSO_SHA256'].pop('candidates/glm-attention-cohort-r9054/libglm_attention_cohort_r9054.so');d['DSO_SHA256']['candidates/glm-M16-cohort-r9056b/libglm_M16_cohort_r9056b.so']=ds;d['matrix'].update(prefix=[16,512],decode=[16,512,1],candidate='SamePattentioncohort M16 only;R128fallbacksamebotharms;twoB8prefixstableID0..15;176requestchecks')
 elif 'DSO_SHA256'in d:d['DSO_SHA256']=ds
 (r/new).write_bytes((json.dumps(d,indent=2)+'\n').encode());files.append(new)
old='observe_GLM_attention_cohort_layer_r9054.py';(r/rename(old)).write_bytes(rename((r/old).read_text()).encode())
for n in files:
 if n.endswith('.py'):ast.parse((r/n).read_bytes())
 assert b'\r'not in(r/n).read_bytes()
print(json.dumps({'files':files,'CPU':proof['producer'],'DSO_SHA256':ds,'M16':True,'no_639_overwrite':True}))
