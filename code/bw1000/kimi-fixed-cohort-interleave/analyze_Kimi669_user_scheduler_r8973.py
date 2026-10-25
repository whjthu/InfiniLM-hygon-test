import pathlib,json,hashlib,statistics as s
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='capacity96service8669aaj-f09f62e8537c5a81';out={'CPU_only':True,'source_handle':'910375.669','nodes':[],'cases':[]}
for node in range(4):
 p=r/'results'/f'kimi-capacity96-service-r8669aaj-{run}-node{node}.json';raw=p.read_bytes();q=json.loads(raw);n={'node':node,'SHA256':hashlib.sha256(raw).hexdigest(),'cases':[]}
 for z in q['serving_cases']:
  if not z['label'].startswith(('B64-','B96-')):continue
  a=[v for v in q['local_forward_samples']if v['label'].startswith(z['label']+'-')];pref=[v for v in a if '-prefill'in v['label']];dec=[v for v in a if '-decode'in v['label']];n['cases'].append({'B':z['B'],'prefill_calls':len(pref),'decode_calls':len(dec),'prefill_CAPI_sum_s':sum(v['CAPI_seconds']for v in pref),'decode_CAPI_sum_s':sum(v['CAPI_seconds']for v in dec),'decode_CAPI_median_s':s.median(v['CAPI_seconds']for v in dec),'decode_CAPI_min_s':min(v['CAPI_seconds']for v in dec),'decode_CAPI_max_s':max(v['CAPI_seconds']for v in dec)})
  if node!=3:continue
  t=z['token_ready_times_from_phase_start'];first=[v[1]-v[0]for v in t];later=[b-a for row in t for a,b in zip(row[1:],row[2:])];sort=sorted(later)
  row={k:z[k]for k in ['B','prefill_wall_seconds','prefill_CAPI_loop_seconds','prefill_final_drain_seconds','anchor_barrier_seconds','decode_wall_seconds','service_wall_seconds']};row.update(first_second_gap_min=min(first),first_second_gap_mean=s.mean(first),first_second_gap_max=max(first),later_ITL_min=min(later),later_ITL_median=s.median(later),later_ITL_p95=sort[int(.95*(len(sort)-1))],later_ITL_max=max(later),maxgap_is_first_second=all(max(b-a for a,b in zip(v,v[1:]))==v[1]-v[0]for v in t),per_16row_cohort=[{'cohort':i//16,'TTFT_first':t[i][0],'second_token':t[i][1],'last_token':t[i][-1],'first_second_gap':first[i],'steady_ITL_median':s.median(b-a for a,b in zip(t[i][1:],t[i][2:]))}for i in range(0,z['B'],16)]);out['cases'].append(row)
 n['feedback']=[x for x in q.get('continuous_feedback_cases',[])if x['label'].startswith(('B64','B96'))];out['nodes'].append(n)
print(json.dumps(out))
