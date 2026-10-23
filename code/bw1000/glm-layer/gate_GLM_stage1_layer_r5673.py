"""Independent CPU-only completeness/math/lifetime and KFD gate."""
import pathlib,json,os,socket,hashlib,time,math,statistics
r=pathlib.Path('/refactor');node=socket.gethostname();step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];reports=[]
paths=sorted((r/'results').glob('glm-stage1-layer-r5673-'+node+'-rank*.json'));assert len(paths)==8
for p in paths:
 q=json.loads(p.read_text());assert q['completed']and q['producer']==step and q['Runtime_weak_expired']and q['normal_reset_close']and q['Torch_postclose']and q['only_one_layer_weight_load']
 assert q['layer']==38 and q['environment']=={'GLM_GROUPED_ROUTED_R5123':'0','GLM_DECODE_ROW_DEDUP_R5386':'0','GLM_BATCH_ALLSELECTED_R5096':'1','GLM_EXACT_BANKS_R4188':'1','GLM_PROJECTED_HOST_R4504':'1'}
 assert len(q['cases'])==3 and len(q['prefill_diagnostic'])==2 and q['free_after_load']>=6*1024**3
 for extra,c in zip([0,63,126],q['cases']):
  assert c['extra_decode_steps']==extra and c['same_initial_history_each_trial']and c['reset_prefill_history_excluded_from_timed_wall']
  assert len(c['profileOFF'])==8 and len(c['profileON_observer_only'])==4 and len(c['numeric_comparisons'])==10
  for rows in [c['profileOFF'],c['profileON_observer_only']]:
   for a in rows:
    assert a['N']==a['B']==8 and len(a['route_histogram'])==256 and sum(a['route_histogram'])==64
    assert all(math.isfinite(a[k])and a[k]>0 for k in ['wall_ms','threadCPU_ms'])and a['route_readback_outside_wall']
    assert a['route_U']==sum(v>0 for v in a['route_histogram'])and a['route_maxrows']==max(a['route_histogram'])
  for a in c['numeric_comparisons']:assert a['max_abs']<=0.03125+0.005*a['reference_max_abs']
 reports.append({'rank':q['rank'],'file':p.name,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert sorted(v['rank']for v in reports)==list(range(8))
kfd=pathlib.Path('/sys/class/kfd/kfd/proc');assert kfd.exists()
for attempt in range(21):
 pids=sorted(x.name for x in kfd.iterdir()if x.name.isdigit())
 if not pids:break
 time.sleep(.25)
assert pids==[],pids
out=r/'results'/('glm-stage1-layer-gate-r5673-'+step+'.json');assert not out.exists();q={'passed':True,'step':step,'reports':reports,'KFD_after_children':pids,'scope':'One synthetic-input actual weighted Decoder layer; profiling source/finite repeat and lifecycle only. No independent GEMM CPU oracle, real Model activations, full tokens or Model TPS.'};out.write_text(json.dumps(q,indent=2));print(json.dumps(q))
