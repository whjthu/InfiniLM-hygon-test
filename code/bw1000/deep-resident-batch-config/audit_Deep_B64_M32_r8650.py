"""Read original terminal only; short negative control never qualifies performance."""
import hashlib,json,math,pathlib,subprocess,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag=sys.argv[1]
rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-24','-j',tag,'-n','-P','-o','JobID,State,ExitCode'],text=True).splitlines()]
assert len(rows)==1 and rows[0][1:]==['COMPLETED','0:0']
p=r/'results'/f'deep-B64-M32-r8650-{tag}.json';q=json.loads(p.read_text());out=p.with_name(p.stem+'-audit.json');assert not out.exists()
assert q['completed']and q['engine_worker_shutdown_completed']and not q['performance_adopted']
m=json.loads((r/'incoming/Deep-B64-M32-config-r8650-manifest.json').read_text())
for n,h in m['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
assert q['library_sha256']==m['DSO_SHA256']
ref=json.loads((r/'incoming/Deep-B32-samegeom-reference-r7269.json').read_text())
assert q['finite_before_passed'];trials=q['config_trials_r8650'];assert trials and trials[0]['batch']==64
assert all(c['micro']==32 for c in trials)
for c in trials:
 B,N=c['batch'],c['requested_outputs'];assert B in [32,64]and N in [32,128]
 if c['complete']:
  assert c['generated_ids']==[ref['generated_ids'][i%32][:N]for i in range(B)]
  assert c['full_IDs_EOS_exact']==(N==128)and c['short_control_only']==(N==32)
  expected=[(0,a,4)for a in range(0,B,4)]+[(s,a,32)for s in range(1,N)for a in range(0,B,32)]
  assert [(v['step'],v['chunk'],v['rows'])for v in c['CAPI_windows']]==expected
  pre=sum(v['seconds']for v in c['CAPI_windows']if v['step']==0);dec=sum(v['seconds']for v in c['CAPI_windows']if v['step']>0)
  assert all(math.isfinite(v['seconds'])and v['seconds']>0 for v in c['CAPI_windows'])
  assert math.isclose(pre,c['native_prefill_seconds'],rel_tol=1e-12)and math.isclose(dec,c['native_decode_seconds'],rel_tol=1e-12)
  if N==128:
   assert math.isclose(B*127/dec,c['native_decode_tok_s'],rel_tol=1e-12)
   assert math.isclose(B*128/c['driver_wall_seconds'],c['driver_E2E_tok_s'],rel_tol=1e-12)
  else:assert 'native_decode_tok_s'not in c and 'driver_E2E_tok_s'not in c
  for i,(ts,metrics)in enumerate(zip(c['request_completion_offsets_seconds'],c['request_metrics'])):
   assert len(ts)==N and all(math.isfinite(x)and x>0 for x in ts)
   gaps=[b-a for a,b in zip(ts,ts[1:])];assert min(gaps)>0
   assert metrics['TTFT_seconds']==ts[0]and math.isclose(metrics['decode_tok_s'],(N-1)/(ts[-1]-ts[0]),rel_tol=1e-12)
   ordered=sorted(gaps);position=.95*(N-2);lo=int(position);p95=ordered[lo]+(position-lo)*(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])
   assert math.isclose(metrics['ITL_p95_seconds'],p95,rel_tol=1e-12)and metrics['max_gap_seconds']==max(gaps)
 else:
  if 'failure'in c:
   f=c['failure'];assert f['CAPI_code']==0
   assert len(c['generated_ids'])==B
   for item in f.get('top2',[]):
    assert item['top1_ID']==f['actual_IDs'][f['rows'].index(item['row'])]
    assert item['reference_ID']==ref['generated_ids'][item['row']%32][f['step']]
    assert math.isclose(item['margin'],item['top1_logit']-item['top2_logit'],rel_tol=1e-12,abs_tol=1e-12)
if any(c['requested_outputs']==32 for c in trials):
 assert trials[0].get('healthy_token_negative')and q['negative_precontrol_health']['last_CAPI_code']==0
 assert trials[1]['batch']==32 and trials[1]['requested_outputs']==32
 assert trials[1]['fresh_cache_reset_seconds']>=0 and not q['performance_qualified']
if q['performance_qualified']:
 assert [c['batch']for c in trials]in [[64,32],[64,32,32,64]]and all(c['complete']and c['requested_outputs']==128 for c in trials)
 assert q['finite_after_restore_passed']and q['quality_passed']
 if len(trials)==4:assert min(q['firstpair_capacity_decode_ratio'],q['firstpair_fullcost_E2E_ratio'])>1.03
for key in ['owner_affinity_before_weight_load','owner_affinity_after_weight_load','owner_affinity']:
 a=q[key];assert a['verified']and a['width']==4
 owners=a['actual_owner_report']['owners'];assert sorted(v['device']for v in owners)==list(range(8))
 masks=[int(x)for v in owners for x in v['verified_mask'].split(',')];assert len(masks)==len(set(masks))==32
result=dict(completed=True,step=tag,normal8owner_shutdown=True,source_report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),outcome=q['outcome'],performance_qualified=q['performance_qualified'],performance_adopted=False,scope=q['scope'],trials=[{k:v for k,v in c.items()if k not in ['generated_ids','CAPI_windows','request_completion_offsets_seconds','request_metrics','per_step_completion_seconds']}for c in trials],no_resubmit=True)
with out.open('x')as f:json.dump(result,f,indent=2)
print(json.dumps(result))
