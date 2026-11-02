"""Original hook output attribution reader; no Native/weights/GPU/new probe."""
from pathlib import Path
import json,hashlib,subprocess,statistics,math,datetime,traceback,sys
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/GLM-layer-critical-path-r370';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(D))
from profile_contract import validate_profile,FIELDS
GROUPS={
 'layer_inclusive':('layer.begin','layer.return'),
 'RMS_attention':('decoder.RMS_attention.begin','decoder.attention.begin'),
 'attention_inclusive':('decoder.attention.begin','decoder.attention.end'),
 'attention_prepare':('attn.prepare.begin','attn.indexer.begin'),
 'metadata_only_selection':('attn.indexer.begin','attn.projectedKV.begin'),
 'attention_expandKV':('attn.projectedKV.begin','attn.cache.begin'),
 'cache_and_row_setup':('attn.cache.begin','attn.row.compute.begin'),
 'row_attention_compute':('attn.row.compute.begin','attn.row.compute.end'),
 'attention_output_inclusive':('attn.output.begin','attn.output.end'),
 'attention_output_quant_and_scale':('attnout.quant.begin','attnout.projection.begin'),
 'attention_output_projection':('attnout.projection.begin','attnout.TP.begin'),
 'attention_TP_copy_allocate_collective':('attnout.TP.begin','attnout.TP.end'),
 'attention_residual':('decoder.attention.end','decoder.RMS_ffn.begin'),
 'RMS_ffn':('decoder.RMS_ffn.begin','decoder.FFN.begin'),
 'FFN_inclusive':('decoder.FFN.begin','decoder.FFN.end'),
 'FFN_router':('decoder.FFN.begin','FFN.router.end'),
 'FFN_input_quant_and_alloc':('FFN.router.end','FFN.gateup.begin'),
 'FFN_gateup':('FFN.gateup.begin','FFN.gateup.end'),
 'FFN_activation_and_alloc':('FFN.gateup.end','FFN.downquant.begin'),
 'FFN_down_quant_and_alloc':('FFN.downquant.begin','FFN.down.begin'),
 'FFN_down':('FFN.down.begin','FFN.down.end'),
 'FFN_weighted_merge_setup':('FFN.down.end','FFN.shared.begin'),
 'FFN_shared_and_routed_merge':('FFN.shared.begin','FFN.shared.end_TP.begin'),
 'FFN_TP_copy_allocate_collective':('FFN.shared.end_TP.begin','FFN.TP.end_cast.begin'),
 'FFN_finalcast_lease_and_return':('FFN.TP.end_cast.begin','decoder.FFN.end'),
 'decoder_final_residual_and_drain':('decoder.FFN.end','layer.return')}
q=dict(completed=False,passed=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Direct038d onecheckpointlayer38 TP8, syntheticprefix512 allvisible; existingprofile0/1 hook diagnostic, not8900/32Kattention or Model cost',CPU_only=True,GPU_initialized=False,Native_dlopen=False,new_GPU_or_probe=False,optimization_gain_claim=False,Model_gain=False)
try:
 path=R/'results/GLM370-watch-r1.json';w=json.loads(path.read_bytes());assert w['completed'];q.update(watch=w,watch_SHA256=sha(path),actual_step=w['actual_step']);step=w['actual_step'];assert step
 raw=subprocess.check_output(['sacct','-S','2026-11-01','-j',step,'-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25);q['accounting']=raw
 if not w['passed']:
  q.update(decision='EXISTING_HOOK_OR_ENGINEERING_FAILURE_STOP_NO_ATTRIBUTION',controller_error=w.get('error'),owner_failures=[])
  for rank in range(8):
   p=R/'results'/('GLM370-'+step+'-rank%d.json'%rank)
   if p.exists():
    a=json.loads(p.read_bytes());q['owner_failures'].append(dict(rank=rank,path=str(p),SHA256=sha(p),error=a.get('error'),traceback=a.get('traceback'),warm=len(a.get('warm',[])),formal=len(a.get('trials',[]))))
 else:
  assert w['normal8']and w['srun_exit0']and w['Slurm_COMPLETED0']and w['post']['passed'];rows=[line.split('|')for line in raw.splitlines()if line.strip()];assert len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0']
  m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes());assert sha(D/'GPU_SOURCE_MANIFEST.json')==w['GPU_source_manifest_SHA256'];owners=[];data={};groupstats=[];raw_adjacent=[]
  for o in w['owners']:
   p=Path(o['path']);assert sha(p)==o['SHA256'];a=json.loads(p.read_bytes());rank=a['rank'];assert a['normal']and a['completed']and a['normal_reset_destroy']and a['Runtime_weak_expired']and a['Torch_after_destroy']and a['measurement_calls']==12 and a['warm_calls']==2 and a['Native_library_SHA256']==m['Native_library_SHA256'];assert len(a['trials'])==12 and len(a['warm'])==2
   for t in a['trials']+a['warm']:
    validate_profile(dict(N=1,B=1,layers=[38],profile=t['profile'],stages=t['stages']),t['mode']);assert t['finite']and t['canary']and t['input_readonly']and t['relative_RMS']<=.005 and t['ownrepeat_relative_RMS']<=.005 and t['vector_counts_delta']==[0,1,0,1,0,0,0,0]
   assert[t['mode']for t in a['trials']]==[0]*3+[1]*6+[0]*3;assert len({(t['arm'],t['rep'])for t in a['trials']})==12
   data[rank]=a;owners.append(dict(rank=rank,path=str(p),SHA256=sha(p),PID=a['PID'],PCI=a['physical_PCI'],weight_load_seconds=a['weight_load_seconds'],free_after_load=a['free_after_load']))
   profile=[t for t in a['trials']if t['profile']==1];names0=[profile[0]['stages'][0]['from']]+[s['to']for s in profile[0]['stages']]
   assert all([t['stages'][0]['from']]+[s['to']for s in t['stages']]==names0 for t in profile)
   for name,(begin,end)in GROUPS.items():
    if names0.count(begin)!=1 or names0.count(end)!=1:groupstats.append(dict(rank=rank,group=name,status='UNKNOWN_missing_or_repeated_endpoint'));continue
    first,last=names0.index(begin),names0.index(end);assert first<last
    values={field:[sum(s[field]for s in t['stages'][first:last])for t in profile]for field in FIELDS}
    groupstats.append(dict(rank=rank,group=name,begin=begin,end=end,samples=6,adjacent_disjoint_same_clock_sum_only=True,nested_groups_not_additive=True,medians_ms={field:statistics.median(v)for field,v in values.items()},ranges_ms={field:[min(v),max(v)]for field,v in values.items()}))
   for index,(begin,end)in enumerate(zip(names0,names0[1:])):
    raw_adjacent.append(dict(rank=rank,index=index,**{'from':begin,'to':end},samples=6,medians_ms={field:statistics.median(t['stages'][index][field]for t in profile)for field in FIELDS}))
  assert set(data)==set(range(8))and len({o['PCI']for o in owners})==8 and all(a['input_fixtures']==data[0]['input_fixtures']for a in data.values())
  aligned=[]
  for arm,mode in enumerate([0,1,1,0]):
   for rep in range(3):
    samples=[next(t for t in a['trials']if t['arm']==arm and t['rep']==rep)for a in data.values()];assert all(t['mode']==mode for t in samples)
    aligned.append(dict(arm=arm,rep=rep,profile=mode,max8_complete_driver_wall_s=max(t['complete_driver_wall_s']for t in samples),max8_native_wall_ms=max(t['whole_layer_native_wall_ms']for t in samples),max8_owner_threadCPU_ms=max(t['owner_threadCPU_ms']for t in samples)))
  overhead={}
  for field in ['max8_complete_driver_wall_s','max8_native_wall_ms','max8_owner_threadCPU_ms']:
   armmean=[statistics.mean(t[field]for t in aligned if t['arm']==arm)for arm in range(4)];off=(armmean[0]+armmean[3])/2;on=(armmean[1]+armmean[2])/2;overhead[field]=dict(arm_means=armmean,OFF_mean=off,ON_mean=on,ON_overhead_fraction=on/off-1,OFF_endpoint_drift_fraction=abs(armmean[3]/armmean[0]-1))
  primary=overhead['max8_complete_driver_wall_s'];low=primary['ON_overhead_fraction']<=m['profile_contract']['maximum_profile_ON_overhead_for_low_overhead_screen']and primary['OFF_endpoint_drift_fraction']<=m['profile_contract']['maximum_absolute_OFF_endpoint_drift_for_quantitative_screen']
  q.update(passed=True,decision='PROFILE_LOW_OVERHEAD_SCREEN_PASS_COMPONENT_ENCLOSURES_ONLY'if low else'PROFILE_EXECUTION_PASS_QUANTITATIVE_ATTRIBUTION_UNKNOWN_DUE_TO_OVERHEAD_OR_DRIFT',low_overhead_screen_passed=low,quantitative_reclaimable_wall='UNKNOWN; profileenclosures are not pureGPUbusy, exclusive kernels or guaranteed reclaimable budget',profile_overhead=overhead,aligned_trials=aligned,owner_group_medians=groupstats,all_original_adjacent_bucket_medians=raw_adjacent,owners=owners,original_raw_tapes_retained_per_owner=True,normal_cleanup=w['post'],all8_normal_and_uniquePCI=True,maximum_relative_RMS=max(t['relative_RMS']for a in data.values()for t in a['trials']+a['warm']),maximum_repeat_RMS=max(t['ownrepeat_relative_RMS']for a in data.values()for t in a['trials']+a['warm']),Native_library_SHA256=m['Native_library_SHA256'],GPU_source_manifest_SHA256=sha(D/'GPU_SOURCE_MANIFEST.json'),numeric_or_lifetime_change_observed=False,interpretation='All rawstage event/host/CPU fields retained; groups only sum contiguous disjoint adjacent intervals within oneownerclock, never sum nestedgroups or crossowner clocks, never Native minus event or CPU as exclusive time',no_new_optimization_or_grid=True,no_longpast_attention_or_Model_projection=True)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['completed']=True;p=R/'results/GLM370-profile-audit-r1.json'
with p.open('x')as f:json.dump(q,f,indent=2)
q['remote_audit_path']=str(p);q['remote_audit_SHA256']=sha(p);print(json.dumps(q))
