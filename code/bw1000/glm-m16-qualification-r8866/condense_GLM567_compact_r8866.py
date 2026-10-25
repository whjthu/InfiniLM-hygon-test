import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent;p=r/'GLM567-M16-qualified-compact-r8866.json';d=json.loads(p.read_bytes());d['expanded_CPU_summary']={'file':p.name,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Same original567 evidence, expanded repeated JSON fields; no extra capture/run'}
columns=['stage','rank','SHA256','normal','completed_arms','completed_arms_gold','beforeQA','afterQA_or_explicit_rejection_scope','load_seconds'];d['owner_report_path_template']='results/glm-transient-Model-r8866-{run}-stage{stage}-rank{rank}.json';d['owner_reports']={'columns':columns,'rows':[[x[k]for k in columns]for x in d['owner_reports']]}
for stage in d['phase_memory_not_peak']:
 stage['columns']=['phase','owners','free_bytes_min','free_bytes_mean','free_bytes_max']
 stage['rows']=[[name,x['owners'],x['free_bytes']['min'],x['free_bytes']['mean'],x['free_bytes']['max']]for name,x in stage.pop('saved_phase_free_not_peak').items()]
d['phase_memory_not_peak']=[{'stage':x['stage'],'recorded_phase_count':len(x['rows']),'recorded_free_min':min(v[2]for v in x['rows']),'recorded_free_max':max(v[4]for v in x['rows']),'scope':'Not peak; every phase/owner summary retained in expanded_CPU_summary plus original phase artifacts'}for x in d['phase_memory_not_peak']]
for arm in d['arms']:
 arm.pop('actual_request_firstlast_TPS')
 arm['all32_per_request_firstlast_values']='Retained in expanded_CPU_summary and actual tailreport; summary here uses all32.'
d['same_pair_decode_gain_percent']=100*(d['B32M16_over_B32M8_decode_ratio']-1);d['same_pair_driver_E2E_speedup_percent']=100*(d['arms'][0]['driver_E2E_s']/d['arms'][1]['driver_E2E_s']-1)
q=r/'GLM567-M16-qualified-compact-r8866-b.json';q.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode());assert q.stat().st_size<16000
print(json.dumps({'file':q.name,'bytes':q.stat().st_size,'SHA256':hashlib.sha256(q.read_bytes()).hexdigest(),'qualified':d['qualified'],'normal':d['normal_owners'],'decode_gain_percent':d['same_pair_decode_gain_percent'],'driver_E2E_speedup_percent':d['same_pair_driver_E2E_speedup_percent']}))
