"""Frozen scalar reader only: no Runtime, GPU, weights, capture or new probe."""
from pathlib import Path
import datetime,hashlib,json,math,statistics,subprocess,traceback
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/Kimi-MoE-prefill512-micro64-r391-gpu-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Geometry-only one current checkpointMoElayer/TP16; same512matrix versus8whole64 calls, allshared/projections/TP/stitch/temporaryrelease/finaldrain enclosed. Not realprefilltraffic/Model/KDA/MLA/capture/service qualification.',CPU_only_reader=True,new_Slurm_or_GPU=False,Native_dlopen=False)
try:
 watch=R/'results/Kimi391-GPU-watch-v1.json';w=json.loads(watch.read_bytes());assert w['completed'];q.update(watch_SHA256=sha(watch),watch=w,actual_step=w['actual_step']);step=w['actual_step'];assert step
 accounting=subprocess.check_output(['sacct','-S','2026-11-01','-j',step,'-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25);q['accounting']=accounting
 if not w['passed']:
  q.update(decision='ENGINEERING_OR_NUMERICAL_FAILURE_STOP_NO_PERFORMANCE_QUALIFICATION',controller_error=w.get('error'),partial_reports=[])
  for node in range(2):
   p=R/'results'/('Kimi391-'+step+'-node%d.json'%node)
   if p.exists():
    a=json.loads(p.read_bytes());q['partial_reports'].append(dict(node=node,path=str(p),SHA256=sha(p),completed=a.get('completed'),phase=a.get('phase'),error=a.get('error')))
   for p in p.with_suffix('').glob('rank*-progress.json'):
    a=json.loads(p.read_bytes());q['partial_reports'].append(dict(node=node,rank=a.get('rank'),path=str(p),SHA256=sha(p),completed=a.get('completed'),error=a.get('error'),completed_strategy_calls=len(a.get('trials',[]))))
 else:
  assert w['normal16']and w['srun_exit0']and w['Slurm_COMPLETED0']and len(w['post'])==2 and all(a['passed']for a in w['post']);lines=[s.split('|')for s in accounting.splitlines()if s.strip()];assert len(lines)==1 and lines[0][0]==step and lines[0][2:4]==['COMPLETED','0:0']
  mf=D/'GPU_SOURCE_MANIFEST.json';assert sha(mf)==w['GPU_source_manifest_SHA256'];m=json.loads(mf.read_bytes());ranks={};owners=[];route_summary=[]
  for expected in w['owners']:
   p=Path(expected['path']);assert sha(p)==expected['SHA256'];a=json.loads(p.read_bytes());assert a['completed']and a['normal_CUDA_cleanup']and a['Torch_after_Runtime_close']and a['library_SHA256']==m['library_SHA256']and a['Native_SHA256']==m['Native_SHA256'];assert a['planned_formal']==96 and a['planned_warm']==6 and len(a['warm_raw_proof'])==48
   owners.append(dict(node=a['node'],path=str(p),SHA256=sha(p),PID=a['PID'],inputs=a['inputs'],weight_load_seconds=a['weight_load_seconds'],CPU_weight_bytes=a['CPU_weight_bytes'],providers=a['providers'],normal_CUDA_cleanup=True))
   assert a['native']['completed']and a['native']['all8normal_before_communicators']and len(a['native']['reports'])==8
   for rank in a['native']['reports']:
    n=rank['global_rank'];assert n not in ranks and rank['completed']and rank['normal_Runtime_close']and rank['module_and_observer_closed'];assert len(rank['trials'])==102
    for t in rank['trials']:
     mode=t['strategy'];assert t['input_readonly']and t['output_tail_canary']and t['counters_pass']and t['error_vs_A']['screen_pass']and t['ownrepeat']['screen_pass']and t['error_vs_A']['finite']and t['ownrepeat']['finite']
     assert t['I192_counts_delta']==([8,512]if mode==2 else[0,0]);assert t['helper_metadata_counts_delta'][0:2]==([1,512]if mode==0 else[0,0]);assert t['legacy_serving_counts_delta']==[0]*4
     assert t['I192_T_hist_delta']==[8 if mode==2 and i==64 else 0 for i in range(65)];assert math.isfinite(t['full_strategy_wall_ms'])and t['full_strategy_wall_ms']>0
    ranks[n]=rank
  assert set(ranks)==set(range(16))and owners[0]['inputs']==owners[1]['inputs']
  key=lambda t:(t['fixture'],t['block'],t['phase'],t['rep']);lookups={n:{key(t):t for t in r['trials']if not t['warm']}for n,r in ranks.items()};assert all(len(v)==96 for v in lookups.values());aligned=[]
  for k in sorted(lookups[0]):
   t=lookups[0][k];assert all(lookups[n][k]['strategy']==t['strategy']for n in range(16));aligned.append(dict(fixture=k[0],block=k[1],phase=k[2],rep=k[3],strategy=t['strategy'],max16_full_strategy_wall_ms=max(lookups[n][k]['full_strategy_wall_ms']for n in range(16))))
  summaries=[]
  for fixture in range(2):
   blocks=[]
   for block in range(4):
    rows={s:[t['max16_full_strategy_wall_ms']for t in aligned if t['fixture']==fixture and t['block']==block and t['strategy']==s]for s in range(3)};assert all(len(v)==4 for v in rows.values());means={s:statistics.mean(rows[s])for s in rows};blocks.append(dict(block=block,A_ms=means[0],B_ms=means[1],C_ms=means[2],C_over_A_rate_gain=means[0]/means[2]-1,C_over_B_rate_gain=means[1]/means[2]-1,B_over_A_rate_change=means[0]/means[1]-1))
   gains=[b['C_over_A_rate_gain']for b in blocks];drift=abs(blocks[-1]['A_ms']/blocks[0]['A_ms']-1);gate=m['performance_gate'];keep=statistics.median(gains)>=gate['minimum_median_C_over_A_rate_gain_per_fixture']and sum(g>0 for g in gains)>=gate['minimum_positive_blocks_per_fixture']and drift<=gate['maximum_A_endpoint_drift'];summaries.append(dict(fixture=fixture,blocks=blocks,median_C_over_A_rate_gain=statistics.median(gains),positive_blocks=sum(g>0 for g in gains),A_endpoint_drift=drift,retained=keep))
   for width in ['T512','eightT64']:
    r=ranks[0]['routing'][str(fixture)][width];equal=all(rank['routing'][str(fixture)][width]==r for rank in ranks.values());route_summary.append(dict(fixture=fixture,width=width,all16_same_router_observer_diagnostic=equal,global_unique_experts=r['global_unique_experts'],global_max_expert_rows=r['global_max_expert_rows'],chunks=r['chunks'],outside_timer=True))
  retained=all(s['retained']for s in summaries);q.update(passed=True,retained=retained,decision='GEOMETRY_COMPONENT_GATE_PASS_MODEL_NOT_QUALIFIED'if retained else'GEOMETRY_COMPONENT_NUMERIC_PASS_PERFORMANCE_GATE_FAIL_STOP',all16_normal=True,owners=owners,aligned_strategy_trials=aligned,summary=summaries,route_observer_summary=route_summary,normal_cleanup=w['post'],CPU_actual='910407.783',CPU_receipt_SHA256=m['CPU_receipt_SHA256'],GPU_source_manifest_SHA256=sha(mf),library_SHA256=m['library_SHA256'],Native_SHA256=m['Native_SHA256'],performance_gate=m['performance_gate'],maximum_relative_RMS=max(t['error_vs_A']['relative_RMS']for r in ranks.values()for t in r['trials']),maximum_ownrepeat_relative_RMS=max(t['ownrepeat']['relative_RMS']for r in ranks.values()for t in r['trials']),Model_gain=False,actual_prefill_activation=False,no_grid_expansion=True,clock='Max16 aligned localelapsed durations; nohostclock subtraction, no overlappingbucket sum or pureGPUbusy inference',B_over_A_is_not_pure_call_overhead=True)
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
q['completed']=True;p=R/'results/Kimi391-terminal-audit-v1.json'
with p.open('x')as f:json.dump(q,f,indent=2)
q['remote_audit_path']=str(p);q['remote_audit_SHA256']=sha(p);print(json.dumps(q))
