from pathlib import Path
import json,hashlib,ast
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cpu=json.loads((r/'GLM-routed-ceiling-CPU-observed-r9046.json').read_bytes());proof=cpu['receipt'];assert proof['completed'];ds=proof['library_SHA256']
def rename(t):return t.replace('lookahead','routed_ceiling').replace('r9044','r9046').replace('glm-gate-routed_ceiling','glm-routed-ceiling')
files=[]
for old in ['GLM_lookahead_layer_r9044.py','GLM_lookahead_layer_r9044_inside.sh','controller_GLM_lookahead_layer_r9044.py','launch_GLM_lookahead_layer_r9044.py']:
 new=rename(old);s=rename((r/old).read_text())
 if old=='GLM_lookahead_layer_r9044.py':
  s=s.replace("DSO_SHA='c1ccce0fddd3967fcda441b5db93068ed689a39c49885122124eea2498eb0077'",'DSO_SHA='+repr(ds))
  s=s.replace('lib.glm_gate_routed_ceiling_policy_r9042.argtypes=[I]','lib.glm_gate_lookahead_policy_r9042.argtypes=[I]\n lib.glm_routed_ceiling_policy_r9046.argtypes=[I]\n lib.glm_routed_ceiling_validate_r9046.argtypes=[ctypes.POINTER(ctypes.c_uint64),Z]')
  s=s.replace('glm_gate_routed_ceiling_counts_r9042','glm_routed_ceiling_counts_r9046').replace('glm_gate_routed_ceiling_policy_r9042','glm_routed_ceiling_policy_r9046')
  s=s.replace('(ctypes.c_uint64*2)();ck(lib.glm_routed_ceiling_counts_r9046(v,2));return [v[0],v[1],0]','(ctypes.c_uint64*3)();ck(lib.glm_routed_ceiling_counts_r9046(v,3));return [v[0]+v[2],v[1],0]')
  s=s.replace("proof['GPU_TUs_compiled_on_CPU']==1 and proof['HOST_TUs_compiled']==0", "proof['GPU_TUs_compiled_on_CPU']==0 and proof['HOST_TUs_compiled']==1")
  s=s.replace("lib=api();assert", "lib=api();ck(lib.glm_gate_lookahead_policy_r9042(0));assert",1)
  s=s.replace("ck(lib.glm_routed_ceiling_policy_r9046(mode));before=stats()", "ck(lib.glm_routed_ceiling_policy_r9046(2 if index==0 else mode));before=stats()")
  needle="actual=decode_out.cpu();after=stats();";assert needle in s
  s=s.replace(needle,"actual=decode_out.cpu();verify_start=time.perf_counter();fixture=(ctypes.c_uint64*3)();ck(lib.glm_routed_ceiling_validate_r9046(fixture,3));fixture_validation_ms=(time.perf_counter()-verify_start)*1000;assert list(fixture)==[1,1,98816],('fixed_fixture_exact',list(fixture));after=stats();")
  # 8x6144 BF16 + 8x8 int64 = 98816 bytes, outside all timing.
  s=s.replace("'wholelayer_native_wall_ms':report['wall_ms']", "'posttimer_fixture_validation_ms':fixture_validation_ms,'fixed_fixture_x_and_ordered_ids_exact':list(fixture)[:2]==[1,1],'posttimer_fixture_readback_bytes':fixture[2],'routed_replay_D2D_bytes':0,'routed_replay_D2D_ms':0,'wholelayer_native_wall_ms':report['wall_ms']")
  s=s.replace("ck(lib.glm_layer_reset_r5670(ctx));expired=I();", "ck(lib.glm_routed_ceiling_policy_r9046(-1));ck(lib.glm_layer_reset_r5670(ctx));expired=I();")
  s=s.replace("try:q['finally_destroy_code']", "try:ck(lib.glm_routed_ceiling_policy_r9046(-1)));q['finally_destroy_code']") if False else s
  s=s.replace("   try:q['finally_destroy_code']=", "   try:ck(lib.glm_routed_ceiling_policy_r9046(-1));q['finally_destroy_code']=")
  s=s.replace('Gate one-block routed_ceiling only; originalgate/down ISA bound; directPV selector alwaysOFF; QK/attention unchanged; finaltokens not qualified','Fixedfixture routed-path zero-copy upper-bound diagnostic; skipgate/silu/mul/downquant/down; router/inputquant/merge/shared/TP/drain unchanged;notproductioncache/Modelgain')
  s=s.replace("'source_input_scope':'Synthetic", "'routed_diagnostic_copy_bytes':0,'routed_diagnostic_copy_ms':0,'source_input_scope':'Synthetic")
 elif old.startswith('controller'):
  s=s.replace("a['finite']and a['canary']", "a['fixed_fixture_x_and_ordered_ids_exact']and a['finite']and a['canary']")
 (r/new).write_bytes(s.encode());files.append(new)
for old in ['GLM-lookahead-layer-manifest-r9044.json','GLM-lookahead-layer-root-GO-r9044.json','GLM-lookahead-layer-execution-authorized-r9044.json']:
 d=json.loads((r/old).read_bytes());new=rename(old)
 if 'sources_SHA256'in d:
  d['sources_SHA256']={n:sha(r/n)for n in files};d.update(scope='FIXEDfixture routedMoE removablewall diagnostic, notoptimization/Modelgain; zero-copy localBF16alias; posttimerexact x/orderedroutechecks',candidate_DSO_SHA256=ds,CPU_handle=proof['producer'],CPU_receipt_SHA256=cpu['receipt_SHA256']);d['DSO_SHA256'].pop('candidates/glm-gate-lookahead-r9044/libglm_gate_lookahead_r9044.so');d['DSO_SHA256']['candidates/glm-routed-ceiling-r9046/libglm_routed_ceiling_r9046.so']=ds;d['matrix']['candidate']='Original vsfixedfixture routedlocal alias;firstwarmcapture only;zeroD2D;router/shared/TPunchanged'
 elif 'DSO_SHA256'in d:d['DSO_SHA256']=ds
 (r/new).write_bytes((json.dumps(d,indent=2)+'\n').encode());files.append(new)
for n in files:
 if n.endswith('.py'):ast.parse((r/n).read_bytes())
 assert b'\r'not in(r/n).read_bytes()
print(json.dumps({'files':files,'DSO_SHA256':ds}))
