"""Offline CPU consumer of the completed 621 capture. Raw data stays remote.

No native initialization, new Model, audit, KFD probe, or performance claim.
"""
import pathlib,json,sys,hashlib,datetime,time,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
sys.path.insert(0,str(r/'incoming'))
from kimi_native_boundary_reader_r8531 import read_boundary_records
from kimi_native_state_consumer_r8519 import summarize_native_target

started=time.perf_counter()
run=json.loads((r/'Kimi-static-localize-r8572-launch.json').read_text())['run']
watch=json.loads((r/'Kimi-static-localize-r8572-watch.json').read_text())
assert watch['actual_handle']=='910375.621' and watch['terminal']['State']=='COMPLETED'
assert watch['terminal']['ExitCode']=='0:0' and watch['post_KFD_passed']
def mapped(value):
 value=str(value);return r/value[len('/refactor/'):]if value.startswith('/refactor/')else pathlib.Path(value)
reports=[]
for node in range(4):
 p=r/'results'/f'kimi-static-localize-r8572-{run}-node{node}.json';q=json.loads(p.read_text())
 assert q['completed'] and q['all8_owned_workers_shutdown_completed']
 reports.append(dict(node=node,SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),report=q))
q=reports[0]['report'];cases=[]
for case in q['diagnostic_cases']:
 source=mapped(case['actuals_path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==case['actuals_SHA256']
 records=json.loads(source.read_text());jobs=[]
 for record in records:
  if record['packed_target']is None:continue
  job=record['job'];ordinal=0 if job['kind']=='prefill'else job['pasts'][job['keys'].index(0)]-511
  boundaries=[];native=[]
  for stage,directory in enumerate(record['perstage_trace_dirs']):
   directory=mapped(directory)
   for value in read_boundary_records(directory):
    assert value['stage']==stage and value['requests'][0]['request']==record['packed_target']
    boundaries.append(value)
   if case['detail_layer']is not None:
    native.extend(summarize_native_target(directory,job,record['packed_target']))
  jobs.append(dict(ordinal0=ordinal,job=job,actual_ids=record['actual_ids'],frozen_ids=record['frozen_ids'],firstdiff=record['firstdiff'],emitter_observation=record['emitter_observation'],boundaries=boundaries,native=native))
 cases.append(dict(label=case['label'],policy=case['policy'],detail_layer=case['detail_layer'],wall_s=case['wall_s'],all_jobs_completed=case['all_jobs_completed'],jobs=jobs))

def index(case):
 out={}
 for job in case['jobs']:
  for value in job['boundaries']:
   key=(job['ordinal0'],value['stage'],value['rank'],value['layer'],'hidden',value['point']);assert key not in out
   req=value['requests'][0];out[key]=dict(hash=req['fnv1a64'],nonfinite=req['nonfinite'])
  for value in job['native']:
   for category in ('states','intermediates'):
    for point,summary in value[category].items():
     key=(job['ordinal0'],value['stage'],value['rank'],value['layer'],category,point);assert key not in out
     out[key]={k:summary[k]for k in ('shape','storage_dtype','bytes','SHA256')}
 return out
pairs=[]
bylabel={v['label']:v for v in cases}
for prefix in ('coarse','late','detail'):
 if prefix+'-O'not in bylabel or prefix+'-N'not in bylabel:continue
 a,b=bylabel[prefix+'-O'],bylabel[prefix+'-N'];ia,ib=index(a),index(b)
 common=sorted(ia.keys()&ib.keys())
 changed=[dict(key=list(k),O=ia[k],N=ib[k])for k in common if ia[k]!=ib[k]]
 pairs.append(dict(prefix=prefix,matched_key_count=len(common),O_only=[list(v)for v in sorted(ia.keys()-ib.keys())],N_only=[list(v)for v in sorted(ib.keys()-ia.keys())],differences=changed))
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_handle='910375.621',run=run,scope='ZERO_NEW_HOOK_STATIC_CAUSAL_INTERVAL_ONLY',performance='N/A',generation_acceptance='FAILED_OBSERVED_618_NOT_RECLASSIFIED',reports=[{k:v[k]for k in ('node','SHA256')}for v in reports],cases=cases,pairs=pairs,localization_decisions=q.get('localization_decisions'),selected_detail_complete=q.get('selected_detail_complete'),detail_budget_exceeded=q.get('detail_budget_exceeded'),actual_snapshot_bytes=q.get('actual_snapshot_bytes'),snapshot_bytes_upper=q.get('snapshot_byte_upper'),limits=['Hash equality/difference is diagnostic only, not proof of first harmful operation or Model acceptance','No residual_slot full array; fullhidden only rank0/6; wholecarry may be UNKNOWN','KDA read-state applicability follows pinned actualcallee8540: positive row read at past0; compare actual read/write row only','MLA compares target logical valid prefix, never unused raw page tails','Teacherforced prefix IDs are actual observations, not autonomous full128/EOS validation','Existing row_output_local/reduced hooks not linked'],all32normal=True,unique_post_KFD=watch['post_KFD'],offline_wall_s=time.perf_counter()-started,raw_storage='REMOTE_ONLY',new_GPU=False,new_Model=False)
p=r/'results'/f'Kimi-static-localize-offline-r8575-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2))
print(json.dumps(dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,scope=out['scope'],pairs=[dict(prefix=v['prefix'],matched=v['matched_key_count'],differences=len(v['differences']))for v in pairs],offline_wall_s=out['offline_wall_s'])))
