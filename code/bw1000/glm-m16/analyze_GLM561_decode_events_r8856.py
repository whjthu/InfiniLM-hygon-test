import pathlib,json,statistics,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run=json.loads((r/'glm-transient-Model-r8855-launch.json').read_text())['run'];out={'CPU_only':True,'GPU':False,'scope':'existing561 ordinary reports only;rounds16..126/T8/past527..637;prefill andwarmdecode excluded;threadCPU maydriver/CCLbusy-poll,notremovablework','owners':[]}
def stat(v):
 v=sorted(v);n=len(v);p=.95*(n-1);i=int(p);return{'n':n,'median':statistics.median(v),'p95':v[i]+(v[min(i+1,n-1)]-v[i])*(p-i),'mean':statistics.mean(v),'min':min(v),'max':max(v)}
for stage in(0,1):
 for rank in range(8):
  p=r/'results'/f'glm-transient-Model-r8855-{run}-stage{stage}-rank{rank}.json';q=json.loads(p.read_bytes());row={'stage':stage,'rank':rank,'report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':[]}
  for c in q['multi_batch_cases']:
   ev=[e for e in c['events']if 16<=e['round']<=126 and e['flat_tokens']==8];assert len(ev)==111*(c['batch']//8)
   walls=[e['native_seconds']for e in ev];cpu=[e['native_owner_thread_cpu_seconds']for e in ev];gaps=[(b['native_start_realtime_ns']-a['native_end_realtime_ns'])/1e9 for a,b in zip(ev,ev[1:])];assert all(x>=0 for x in gaps)
   span=(ev[-1]['native_end_realtime_ns']-ev[0]['native_start_realtime_ns'])/1e9
   row['cases'].append({'B':c['batch'],'M':c['micro'],'decode_calls_total':sum(e['round']>0 for e in c['events']),'decode_calls_selected':len(ev),'native_ms':stat([x*1000 for x in walls]),'threadCPU_ms':stat([x*1000 for x in cpu]),'outside_CAPI_sameowner_gap_ms':stat([x*1000 for x in gaps]),'sum_native_s':sum(walls),'sameowner_realtime_selected_span_s':span,'native_wall_fraction_of_selected_span':sum(walls)/span,'shape_set':sorted({(e['flat_tokens'],len(e['rows']))for e in ev}),'vector_gate_down_hit_rows':c['resident_phase_after']if 'resident_phase_after'in c else None})
  out['owners'].append(row)
print(json.dumps(out))
