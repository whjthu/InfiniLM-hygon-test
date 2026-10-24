"""Original498 audit metadata and existing timing arithmetic only; no audit rerun."""
import datetime,hashlib,json,pathlib,statistics
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
run='dual-5836-6206f352aae344268ed1'
ap=r/'results/GLM-dual-Model-terminal-audit-r5837.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ap)=='777dadb411e5bb2ab3f9570b9c18f1ea64b34a55fa6322b53ac40538270e3ba7'
a=json.loads(ap.read_text());assert a['run']==run and a['producer']=='910376.498'and a['qualified']
wp=r/'results/GLM-dual-Model-watch-r5837.json';w=json.loads(wp.read_text());assert w['run']==run and w['qualified']and w['audit_SHA256']==sha(ap)
launch=json.loads((r/'glm-dual-Model-r5837-launch.json').read_text())
out={'producer':'910376.498','run':run,'read_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_audit_SHA256':sha(ap),'watch_SHA256':sha(wp),'watch':w,'launch_UTC':launch['utc'],'source_manifest_SHA256':launch['source_manifest_SHA256'],'audit':{k:a[k]for k in ['qualified','Slurm','allowner_report_SHA256','preengine_records','network_logs','raw_CPUargmax','common_clock_SHA256','common_clock_completed','common_phases','dual_comparison','candidate_rejected','selected_counts','scope']}}
out['owners']=[{k:o[k]for k in ['replica','stage','rank','normal_closed','pinned_host_bytes','load_wall_seconds']}for o in a['owners']]
assert len(out['owners'])==32 and len({(x['replica'],x['stage'],x['rank'])for x in out['owners']})==32
def distribution(values):
 v=sorted(values)
 def quant(p):
  i=(len(v)-1)*p;j=int(i);return v[j]+(v[min(j+1,len(v)-1)]-v[j])*(i-j)
 return {'count':len(v),'median':statistics.median(v),'P95':quant(.95),'P99':quant(.99),'min':v[0],'max':v[-1],'mean':statistics.mean(v)}if v else None
out['tail_cases']=[]
for replica in(0,1):
 p=r/'results'/f'glm-transient-Model-r5837-{run}-replica{replica}-stage1-rank0.json'
 q=json.loads(p.read_text());assert sha(p)==a['allowner_report_SHA256'][p.name]
 for c in q['multi_batch_cases']:
  times=c['tail_ready_times'];gaps=[b-a for t in times for a,b in zip(t,t[1:])]
  out['tail_cases'].append({'replica':replica,**{k:c[k]for k in ['phase','tail_ready_metrics','prefill_input_tokens_per_service_second','tail_prefill_CAPI_return_seconds','elapsed_seconds','same_micro_reference_passed','cross_geometry_full_IDs_passed','shared_validation_counter_delta','shared_validation_expected','shared_validation_source_exact_avoided_D2H_bytes']},'ITL_seconds':distribution(gaps),'first_native_start_realtime_ns':min(e['native_start_realtime_ns']for e in c['events']),'last_native_end_realtime_ns':max(e['native_end_realtime_ns']for e in c['events'])})
cp=r/'results'/('glm-dual-clock-r5837-'+run+'.json');clock=json.loads(cp.read_text());assert sha(cp)==a['common_clock_SHA256']
out['collector_ITL']=[{'phase':p['label'],'ITL_seconds':distribution([b['receipt_seconds']-a['receipt_seconds']for rows in p['receipt_rows'].values()for h in rows for a,b in zip(h,h[1:])])}for p in clock['completed_phases']]
print(json.dumps(out))
