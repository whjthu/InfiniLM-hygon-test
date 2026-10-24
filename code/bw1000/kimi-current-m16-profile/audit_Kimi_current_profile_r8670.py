"""Terminal-only CPU audit of instrumented current B64M16 profile; no Engine/KFD."""
import pathlib,json,hashlib,math,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
run=json.loads((r/'Kimi-current-profile-r8670-launch.json').read_text())['run']
w=json.loads((r/'Kimi-current-profile-r8670-watch.json').read_text())
out=dict(run=run,scope='Current183f6f B64M16 servingpolicy1 grouped4, two natural512 prompt families repeated; instrumented baseline, not unwrapped throughput or optimization gain',terminal=w['terminal'],post_KFD_passed=w['post_KFD_passed'],owners=[],all32normal=False,qualified=False)
try:
 reports=[];labels=['B64-M16-full-reference128','B64-M16-ordinary0','B64-M16-ordinary1','B64-M16-hiptrace36']
 for node in range(4):
  p=r/'results'/f'kimi-current-profile-r8670-{run}-node{node}.json';q=json.loads(p.read_text());reports.append(q)
  out['owners'].append(dict(node=node,normal8=q.get('all8_owned_workers_shutdown_completed',False),completed=q['completed'],report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),error=q.get('error'),shutdown_error=q.get('shutdown_error')))
  assert q['node']==node and q['PP_stage']==node and q['nodes']==4 and q['TP']==8 and q['PP']==4
  assert q['cache_geometry']==dict(max_rows=64,pages=260,page_size=256,KDA_pool_rows=65,pages_per_request=4,max_total_tokens=1024)
  assert q['library_sha256']=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
  assert q['serving_protocol_complete'] and not q.get('error') and not q.get('shutdown_error')
  cases=[v for v in q['serving_cases']if v['label'].startswith('B64-')]
  assert [v['label']for v in cases]==labels
  assert all(v['B']==64 and v['frontend_cohort_M']==16 and v['policy']==1 and v['matched_baseline_complete_IDs']is True for v in cases)
  assert all(v['generated_ids']==reports[0]['serving_cases'][next(i for i,x in enumerate(reports[0]['serving_cases'])if x['label']==v['label'])]['generated_ids']for v in cases)
  assert len(q['profiling_windows'])==1
  t=q['profiling_windows'][0];assert (t['start_decode_step0'],t['start_cohort'],t['stop_decode_step0'],t['stop_cohort'])==(32,0,33,3)
  assert t['start_returncode']==t['stop_returncode']==0 and t['stop_realtime_ns']>t['start_realtime_ns']
  assert t['start_pasts']==[544]*16,('decode32_before_call_past',t['start_pasts'])
  assert {v['epoch']for v in q['QA']}=={'before','after'} and all(v['complete_frozen_IDs_exact']and v['policy']==0 for v in q['QA'])
  assert all(v['delta']==[v['expected']]*8 for v in q['counter_dispatch'])
  assert any(v['policy']==1 and v['N']==16 and v['expected'][0]>0 and v['expected'][1]==16*v['expected'][0] and v['expected'][2:]==[0,0]for v in q['counter_dispatch'])
  trace=r/'results'/f'kimi-current-profile-trace-r8670-{run}-node{node}'
  files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size)for p in sorted(trace.rglob('*'))if p.is_file()]
  assert files and any(v['bytes']>0 for v in files)
  out.setdefault('trace_artifacts',[]).append(dict(node=node,files=files))
 out['all32normal']=all(v['normal8']and v['completed']for v in out['owners'])
 assert out['all32normal'] and w['post_KFD_passed'] and w['terminal']['State']=='COMPLETED'and w['terminal']['ExitCode']=='0:0'
 q=reports[3];ordinary=[v for v in q['serving_cases']if '-ordinary'in v['label']]
 assert len(ordinary)==2
 for v in ordinary:
  n=sum(sum(x not in set(json.loads((r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_text())['eos_token_ids'])for x in row[1:])for row in v['generated_ids'])
  assert math.isclose(n/v['decode_wall_seconds'],v['aggregate_decode_tok_s'],rel_tol=1e-12)
 out['instrumented_ordinary']=[{k:v[k]for k in ['label','aggregate_decode_tok_s','decode_tok_s_per_card','decode_wall_seconds','prefill_wall_seconds','end_to_end_output_tok_s','TTFT_seconds','per_request_first_last_tok_s','per_request_max_ITL_seconds','instrumented_baseline','user_clock']}for v in ordinary]
 out['loading_seconds_by_node']=[v.get('loading_seconds')for v in reports]
 handle=w['terminal']['JobID'];logs=[]
 for node in range(4):
  matches=list((r/'logs').glob(f'Kimi-current-profile-{run}-{handle}-{node}.log'));assert len(matches)==1,(node,matches)
  data=matches[0].read_text(errors='replace');assert 'Using network IBext_v8'in data and 'Using network Socket'not in data
  bad=[line for line in data.splitlines()if re.search(r'rocBLAS.*(?:error|709)|Traceback \(most recent|HIP error|Segmentation fault|terminate called',line,re.I)]
  assert not bad,('strict_error',node,bad[:8]);logs.append(dict(node=node,path=str(matches[0].relative_to(r)),SHA256=hashlib.sha256(matches[0].read_bytes()).hexdigest(),IBext_v8=True,Socket_fallback=False))
 out['network_logs']=logs;out['qualified']=True
except BaseException as exc:out['error']=repr(exc);raise
finally:
 p=r/'results'/f'Kimi-current-profile-r8670-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2))
 print(json.dumps(dict(run=run,qualified=out['qualified'],all32normal=out['all32normal'],error=out.get('error'))))
