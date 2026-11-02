"""Read original component outputs in place; no Runtime/weights/GPU or new probe."""
from pathlib import Path
import json,hashlib,subprocess,statistics,math,datetime,traceback
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/GLM-smallM-FFN-feasibility-r367';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Component only: onecheckpointlayer38 TP8, synthetic prefix512 and2changedinputs; profileOFF complete layer CAPI includes fullFFN/shared/TP/routeD2H/finalsync, no Model/benchmark/SLO gain',CPU_only_reader=True,GPU_initialized=False,Native_dlopen=False,new_GPU_or_probe=False)
try:
 watch=R/'results/GLM367-watch-r1.json';w=json.loads(watch.read_bytes());assert w['completed'];q.update(watch=w,watch_SHA256=sha(watch),actual_step=w['actual_step'])
 step=w['actual_step'];assert step
 raw=subprocess.check_output(['sacct','-S','2026-11-01','-j',step,'-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25);q['accounting']=raw
 if not w['passed']:
  q.update(decision='ENGINEERING_FAILURE_STOP_NO_PERFORMANCE_QUALIFICATION',controller_error=w.get('error'),owner_failures=[])
  for rank in range(8):
   p=R/'results'/('GLM367-'+step+'-rank%d.json'%rank)
   if p.exists():
    a=json.loads(p.read_bytes());q['owner_failures'].append(dict(rank=rank,path=str(p),SHA256=sha(p),normal=a.get('normal'),completed=a.get('completed'),error=a.get('error'),traceback=a.get('traceback'),Native_error_failstop_no_reset_or_replay=a.get('Native_error_failstop_no_reset_or_replay'),warm_calls_completed=len(a.get('warm',[])),measured_calls_completed=len(a.get('trials',[]))))
 else:
  assert w['normal8']and w['srun_exit0']and w['Slurm_COMPLETED0']and w['post']['passed'];rows=[line.split('|')for line in raw.splitlines()if line.strip()];assert len(rows)==1 and rows[0][0]==step and rows[0][2:4]==['COMPLETED','0:0']
  owners=[];byrank={};manifest=D/'GPU_SOURCE_MANIFEST.json';m=json.loads(manifest.read_bytes());assert sha(manifest)==w['GPU_source_manifest_SHA256']
  for expected in w['owners']:
   p=Path(expected['path']);assert sha(p)==expected['SHA256'];a=json.loads(p.read_bytes());rank=a['rank'];assert a['normal']and a['completed']and a['normal_reset_destroy']and a['Runtime_weak_expired']and a['Torch_after_destroy']and a['policy_closed']and a['profile']==0 and a['measurement_calls']==48 and a['warm_calls']==4 and a['candidate_library_SHA256']==m['candidate_library_SHA256']
   assert len(a['trials'])==48 and len(a['warm'])==4 and a['input_fixtures'][0]['decode_SHA256']!=a['input_fixtures'][1]['decode_SHA256']
   key=lambda r:(r['fixture'],r['pair'],r['arm'],r['rep']);assert len({key(r)for r in a['trials']})==48
   for r in a['trials']+a['warm']:
    mode=r['mode'];assert r['finite']and r['canary']and r['input_readonly']and r['relative_RMS']<=.005 and r['ownrepeat_relative_RMS']<=.005 and r['profile']==0 and r['route_U']==8 and r['route_maxrows']==1
    assert r['M1_counts_delta']==[mode,1-mode,0,0]and r['vector_counts_delta']==[mode,1-mode,0,1,0,mode,0,0]
    assert all(math.isfinite(r[k])and r[k]>0 for k in ['complete_driver_wall_s','whole_layer_native_wall_ms','owner_threadCPU_ms'])
   byrank[rank]=a;owners.append(dict(rank=rank,path=str(p),SHA256=sha(p),PID=a['PID'],PCI=a['physical_PCI'],load_seconds=a['weight_load_seconds'],checkpoint_payload_referenced_bytes=a['checkpoint_payload_referenced_bytes'],free_after_load=a['free_after_load'],free_after_release=a['free_after_release']))
  assert set(byrank)==set(range(8))and len({o['PCI']for o in owners})==8
  assert all(a['input_fixtures']==byrank[0]['input_fixtures']for a in byrank.values())
  lookups={rank:{(r['fixture'],r['pair'],r['arm'],r['rep']):r for r in a['trials']}for rank,a in byrank.items()};aligned=[];summary=[]
  for key in sorted(lookups[0]):
   r=lookups[0][key];assert all(lookups[rank][key]['mode']==r['mode']for rank in range(8))
   aligned.append(dict(fixture=key[0],pair=key[1],arm=key[2],rep=key[3],mode=r['mode'],max8_complete_driver_wall_s=max(lookups[rank][key]['complete_driver_wall_s']for rank in range(8)),max8_whole_layer_native_wall_ms=max(lookups[rank][key]['whole_layer_native_wall_ms']for rank in range(8)),max8_owner_threadCPU_ms=max(lookups[rank][key]['owner_threadCPU_ms']for rank in range(8))))
  for fixture in range(2):
   pairs=[];off=[];on=[]
   for pair in range(4):
    grouped={mode:[r for r in aligned if r['fixture']==fixture and r['pair']==pair and r['mode']==mode]for mode in [0,1]};assert all(len(v)==3 for v in grouped.values())
    means={mode:statistics.mean(r['max8_complete_driver_wall_s']for r in grouped[mode])for mode in [0,1]};off.append(means[0]);on.append(means[1]);pairs.append(dict(pair=pair,OFF_complete_wall_s=means[0],ON_complete_wall_s=means[1],rate_gain=means[0]/means[1]-1))
   changes=[p['rate_gain']for p in pairs];drift=abs(off[-1]/off[0]-1);gate=m['performance_gate'];retained=sum(v>0 for v in changes)>=gate['minimum_positive_pairs_per_input']and statistics.median(changes)>=gate['minimum_median_rate_gain_per_input']and drift<=gate['maximum_OFF_endpoint_drift']
   summary.append(dict(fixture=fixture,pairs=pairs,median_paired_rate_gain=statistics.median(changes),positive_pairs=sum(v>0 for v in changes),OFF_endpoint_drift=drift,OFF_mean_s=statistics.mean(off),ON_mean_s=statistics.mean(on),retained=retained))
  q.update(passed=True,decision='COMPONENT_GATE_PASS_MODEL_NOT_QUALIFIED'if all(r['retained']for r in summary)else'COMPONENT_NUMERIC_PASS_PERFORMANCE_GATE_FAIL_STOP',qualified_math_and_state=True,retained=all(r['retained']for r in summary),owners=owners,aligned_trials=aligned,summary=summary,performance_gate=m['performance_gate'],all8_normal=True,normal_cleanup=w['post'],CPU_actual='910407.726',CPU_receipt_SHA256=m['CPU_receipt_SHA256'],GPU_manifest_SHA256=sha(manifest),candidate_library_SHA256=m['candidate_library_SHA256'],baseline_Native_SHA256=m['baseline_Native_SHA256'],all_warm_and_formal_raw_artifacts_remain_on_server=True,maximum_relative_RMS=max(r['relative_RMS']for a in byrank.values()for r in a['trials']+a['warm']),maximum_ownrepeat_RMS=max(r['ownrepeat_relative_RMS']for a in byrank.values()for r in a['trials']+a['warm']),clock='Same aligned trial max of8 local elapsed durations, no cross-host clocks/overlap sums; Native and driver wall enclosures not pure GPUbusy',pure_GPU_event_time='Not collected; profileOFF required',Model_gain=False,online_or_gt15_qualified=False,no_grid_expansion=True)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['completed']=True
p=R/'results/GLM367-terminal-audit-r1.json'
with p.open('x')as f:json.dump(q,f,indent=2)
q['remote_audit_path']=str(p);q['remote_audit_SHA256']=sha(p);print(json.dumps(q))
