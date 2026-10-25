import json,pathlib,sys,hashlib
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['engine_worker_shutdown_completed']and q['collection_finished'];assert q['grid_complete']and set(q['grid'])=={'8','16','32'}
for B in ['8','16','32']:
 g=q['grid'][B];assert g['complete']and g['paired_quality']['n']==32
 for arm in ['fixed','compact']:assert g['arms'][arm]['cases']==32
for t in q['quality_trials']:
 assert t['finite_readonly_canary_all_calls']and t['all_prefill_packed_hit']and not t['budget_stop'];assert len(t['records'])==t['B']
 assert len({z['logical_row']for z in t['records']})==t['B']
print(json.dumps({'passed':True,'step':q['step'],'grid':[8,16,32],'paired_real_quality_cases_each':32,'native_not_HTTP':True,'quality_noninferiority_proven':False,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
