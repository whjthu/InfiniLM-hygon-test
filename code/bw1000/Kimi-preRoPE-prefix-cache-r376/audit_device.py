"""Recompute one fixed ABBA from sealed scalar reports; never substitutes Model costs."""
from pathlib import Path
import argparse,json,hashlib,gzip,base64,collections,statistics

SHA=lambda b:hashlib.sha256(b).hexdigest()
def audit(raw):
 r=raw['report']['data'];w=raw['watch']['data'];seq=[v['data']for v in raw['sequences']]
 assert r['completed']and r['passed']and w['completed']and w['passed']
 assert w['report_SHA256']==raw['report']['SHA256']
 assert r['actual_step']=='910371.703'and w['CPU_keeper']=='910407.760'
 assert w['srun_exit_code']==0 and w['terminal'][2:4]==['COMPLETED','0:0']
 assert len(seq)==18 and [s['ordinal']for s in seq]==list(range(18))
 assert sum(len(s['queries'])for s in seq)==144
 commands=collections.Counter(v['cmd']for s in seq for v in s['commands'])
 assert dict(commands)==dict(begin=18,append_prompt=498,seal=18,query=144,append_committed=126,retire=18,close=18)
 assert all(s['passed']and s['normal_session_close']and s['completed']for s in seq)
 assert r['normal_runtime_close']and r['original_model_module_parameter_identity_unchanged']
 assert len(r['memory_fences'])==912 and all(f['floor_passed']and f['floor_bytes']==4<<30 for f in r['memory_fences'])
 assert not w['post']['kfd_pids']and not w['post']['matching_component_processes']and all(w['post']['ports_bindable'].values())
 for a,b in zip(raw['sequences'],r['sequence_artifacts']):assert a['SHA256']==b['SHA256']
 numeric=[];timing=[]
 for L in [1024,8635,32638]:
  group=[s for s in seq if s['L']==L];formal=[s for s in group if s['phase']=='formal']
  assert [s['cache']for s in formal]==[False,True,True,False]
  row=dict(L=L,formal_ordinals=[s['ordinal']for s in formal],comparisons={})
  for field in ['continuous_checked_8cycle_wall_s','engine_dispatch_8cycle_sum_s','query_dispatch_sum_s','accepted_append_dispatch_sum_s','prefix_begin_append_seal_wall_s','retire_close_wall_s','total_with_prefix_numeric_and_close_wall_s']:
   v=[s[field]for s in formal];off=(v[0]+v[3])/2;on=(v[1]+v[2])/2
   row['comparisons'][field]=dict(ABBA_s=v,OFF_mean_s=off,ON_mean_s=on,ON_over_OFF=on/off,change_percent=100*(on/off-1),OFF_endpoint_drift_percent=100*(v[3]/v[0]-1))
  row['query_cycles']=[]
  for cycle in range(8):
   v=[s['queries'][cycle]['complete_query_dispatch_wall_s']for s in formal]
   row['query_cycles'].append(dict(cycle=cycle,past=L+8*cycle,ABBA_s=v,OFF_mean_s=(v[0]+v[3])/2,ON_mean_s=(v[1]+v[2])/2))
  row['firstprocess_shape_warm_sequences']=[dict(ordinal=s['ordinal'],cache=s['cache'],prefix_s=s['prefix_begin_append_seal_wall_s'],checked_8cycle_s=s['continuous_checked_8cycle_wall_s'],dispatch_8cycle_s=s['engine_dispatch_8cycle_sum_s'],total_s=s['total_with_prefix_numeric_and_close_wall_s'])for s in group if s['phase']=='warm']
  timing.append(row)
  for enabled in [False,True]:
   queries=[q for s in group if s['cache']==enabled for q in s['queries']]
   allvalues=[q[k]for q in queries for k in ['hidden','logits']]+[v for q in queries for v in q['stages'].values()]
   assert all(q['canary_passed']and q['completed']for q in queries)and all(v['finite']for v in allvalues)
   n=dict(L=L,cache=enabled,queries=len(queries),all_finite=True,all_canaries=True,
    max_hidden_NRMSE=max(q['hidden']['NRMSE']for q in queries),max_logits_NRMSE=max(q['logits']['NRMSE']for q in queries),
    max_stage_NRMSE=max(v['NRMSE']for q in queries for v in q['stages'].values()),
    hidden_above_02_alert_count=sum(q['hidden']['above_02_alert']for q in queries),logits_above_02_alert_count=sum(q['logits']['above_02_alert']for q in queries),
    stage_above_02_alert_count=sum(v['above_02_alert']for q in queries for v in q['stages'].values()),
    proposal_position_matches=sum(q['proposal_position_matches']for q in queries),proposal_positions=7*len(queries),minimum_matches_per_query=min(q['proposal_position_matches']for q in queries),
    first_full_query_hidden_NRMSE=[q['hidden']['NRMSE']for q in queries if q['cycle']==0],
    first_full_query_logits_NRMSE=[q['logits']['NRMSE']for q in queries if q['cycle']==0])
   if enabled:
    assert all(q['cache']['computed_rows']==([L+8]if q['cycle']==0 else [16])for q in queries)
    n['cache_first_full_then_new_ctx8_plus_draft8']=True
   numeric.append(n)
 return dict(schema=1,completed=True,execution_lifecycle_PASS=True,decision='STOP_THIS_IMPLEMENTATION_NO_NET_WHOLE_SEQUENCE_GAIN',
  actual_GPU='910371.703',actual_CPU_keeper='910407.760',source_publication_commit=w['publication_commit'],GPU_manifest_SHA256=r['source_manifest_SHA256'],actual_device=r['actual_device'],torch_runtime=r['torch_runtime'],fixture512_SHA256=r['fixture512_SHA256'],profile=False,target_cards=0,worker_manifest_SHA256=r['worker_manifest_SHA256'],
  original_artifacts=dict(report={k:v for k,v in raw['report'].items()if k!='data'},watch={k:v for k,v in raw['watch'].items()if k!='data'},sequences=[{k:v for k,v in s.items()if k!='data'}for s in raw['sequences']]),
  Slurm_accounting=raw['accounting'],srun_exit_code=0,normal18_contexts=True,normal_Runtime_close=True,original_model_parameter_module_identity_unchanged=True,post=w['post'],
  counts=dict(contexts=18,queries=144,warm_queries=48,formal_queries=96,commands=dict(commands),total_commands=sum(commands.values()),original_4GiB_fences=912),
  paired_timings=timing,numeric_diagnostics=numeric,
  memory=dict(peak_allocated_Torch_bytes=r['peak_allocated_draft_bytes'],peak_reserved_Torch_bytes=r['peak_reserved_draft_bytes'],minimum_observed_before_trim_free_bytes=r['minimum_observed_free_draft_bytes'],minimum_after_floor_free_bytes=r['minimum_free_draft_bytes'],trim_records=[f for f in r['memory_fences']if f['trimmed']],scope='Torch allocator tracked peaks, not whole device peak. All912 original4GiB fences passed.'),
  cost=dict(load_wall_s=r['load_wall_s'],fixture_CPU_generation_wall_s=r['CPU_fixture_generation_wall_s'],total_driver_wall_s=r['total_driver_wall_s'],all_sequence_totals_s=sum(s['total_with_prefix_numeric_and_close_wall_s']for s in seq),warm_sequence_totals_s=sum(s['total_with_prefix_numeric_and_close_wall_s']for s in seq if s['phase']=='warm'),formal_sequence_totals_s=sum(s['total_with_prefix_numeric_and_close_wall_s']for s in seq if s['phase']=='formal'),allocated_GPU_cardseconds_from_Slurm=73,allocated_CPU_keeper_wall_s=82),
  limits=['Single fixed ABBA per geometry; both full checked8cycle and complete dispatch are slower with cache. No unchanged-source rerun or target load.',
   'Checked8cycle includes8queries+7syntheticaccepted8 appends+pending/cat/fences plus canary/stageCPUreadback/inline numeric/JSONsave overhead. Dispatch also contains hooks/cache diagnostics; neither is observerfree production throughput.',
   'Per-query full hidden/logits and actual7proposals saved as scalar diagnostics. Stage samples cover3contextrows+8draft positions; no fullKV equality claim.',
   '.02 alerts and .10 investigationstop are local diagnostic policies, not Modelquality boundaries. Token agreement is diagnostic. Actualtargetacceptance/benchmarkquality/Modelgain UNKNOWN.',
   'First fullM cache query uses originalLinear shapes; later incrementals change GEMM M. Firstquery differences and every later difference retained. SelectedSDPA algorithm and physicalGPUcold UNKNOWN.',
   'All18 prefix/warm/load/retireclose and full residency included separately; no emitted token timeline or online service in this component card.',
   'No rawactivation serialized; numerical summaries cannot independently rederive tensors. Old369 .02failure remains unchanged. No claim that cache mechanism mathematically impossible or other implementation disproven.'])

def main():
 p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True);source.add_argument('--raw');source.add_argument('--sealed-input');p.add_argument('--out',required=True);p.add_argument('--sealed',required=True);a=p.parse_args()
 if a.raw:data=Path(a.raw).read_bytes()
 else:
  original=json.loads(Path(a.sealed_input).read_bytes());data=gzip.decompress(base64.b64decode(original['payload']));assert SHA(data)==original['uncompressed_SHA256']and len(data)==original['uncompressed_bytes']
 raw=json.loads(data);result=audit(raw)
 # Preserve original scalar JSON bytes when reconstruction is exactly SHA-identical.
 for item in [raw['report'],raw['watch']]+raw['sequences']:
  encoded=(json.dumps(item['data'],indent=2)+'\n').encode();assert SHA(encoded)==item['SHA256']
 seal=dict(schema=1,content='Scalar-only original18sequence/report/watch/Slurm/log snapshot. No activation or weight.',uncompressed_SHA256=SHA(data),uncompressed_bytes=len(data),encoding='base64(gzip)',payload=base64.b64encode(gzip.compress(data,mtime=0)).decode())
 Path(a.sealed).write_text(json.dumps(seal,indent=2)+'\n',encoding='utf8');result['sealed_snapshot']=dict(path=str(Path(a.sealed).name),SHA256=SHA(Path(a.sealed).read_bytes()),uncompressed_SHA256=SHA(data))
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 print(json.dumps(dict(completed=True,decision=result['decision'],timing=[dict(L=x['L'],checked_percent=x['comparisons']['continuous_checked_8cycle_wall_s']['change_percent'],dispatch_percent=x['comparisons']['engine_dispatch_8cycle_sum_s']['change_percent'])for x in result['paired_timings']],out=a.out)))
if __name__=='__main__':main()
