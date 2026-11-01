from pathlib import Path
import json,hashlib,math,statistics,sys,subprocess,datetime
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='GLM-C2M1-r355-a';S=R/'incoming/GLM-two-cohort-Model-r355/owner';C=R/'incoming/GLM-two-cohort-Model-r355/controller';pin='32f24ae36f4148c9eac9f86294a4bcccb3c1f269d6b1123cb9750d88a0693366';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Resolve the actual original controller/owner using the final immutable watch.
wpath=R/'results'/('GLM-r355-watch-'+run+'.json');w=json.loads(wpath.read_bytes());assert w['completed']and w['passed']and w['post_empty']
assert sha(S/'SOURCE_MANIFEST.json')==pin
m=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
sys.path.insert(0,str(S));from workload_r355 import PLAN;from workload_r349 import validate_reply,body_and_gold
card=json.loads((S/'FIXTURE_CARD.json').read_bytes());fixtures=[]
for original in card['fixtures']:
 f={k:str(R/Path(v).relative_to('/refactor'))if isinstance(v,str)and v.startswith('/refactor/')else v for k,v in original.items()}
 raw=Path(f['request_external_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==f['request_SHA256'];body=json.loads(raw)
 assert body['model']=='GLM-5.3' and body['max_tokens']==4096 and body['temperature']==0 and body['stream'] is False
 if f['id']=='short-read':gold={'path':'input.json'}
 else:
  data=json.loads(next(x['content']for x in body['messages']if x['role']=='tool'and'records'in x['content']));values={x['id']:x['value']for x in data['records']};aa,bb,cc,dd=[values[k]for k in data['query_ids']];assert all(type(x)is int for x in [aa,bb,cc,dd]);gold={'a':aa,'b':bb}
 encoded=Path(f['current_prepared_prompt_IDs']).read_bytes();assert hashlib.sha256(encoded).hexdigest()==f['current_prepared_private_IDs_file_SHA256'];ids=json.loads(encoded)['prompt_token_ids']
 assert len(ids)==f['current_prepared_prompt_tokens']and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==f['current_prepared_prompt_IDs_SHA256']
 fixtures.append(dict(fixture=f,ids=ids,gold=gold))
assert[len(f['ids'])for f in fixtures]==[479,8900]
signature=None;owners=[];tail=None
for rec in w['evidence']['owner_reports']:
 p=Path(rec['path']);assert sha(p)==rec['SHA256'];q=json.loads(p.read_bytes())
 assert all(q[k]for k in ['completed','worker_shutdown_completed','normal_reset_close','normal_shutdown','observer_native_destroy_returned0','post_close_Torch_liveness'])
 assert q['source_manifest_SHA256']==pin and q['library_sha256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
 assert q['constructor_rows']==2 and len(q['pairs'])==6
 sig=[]
 for i,pair in enumerate(q['pairs']):
  assert pair['index']==i and pair['enabled']==PLAN[i]and pair['warmup']==(i<2)
  for row,r in enumerate(pair['requests']):
   ids=r['generated_ids'];assert len(ids)==r['tokens']and ids[-1]in[154820,154827,154829]and r['natural_EOS']and r['normal_reset_all32']
   assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==r['generated_IDs_SHA256']
   assert r['prompt_IDs_SHA256']==fixtures[row]['fixture']['current_prepared_prompt_IDs_SHA256']
   calls=[c for c in pair['calls']if c['row']==row];assert sum(c['T']for c in calls if c['prefill'])==len(fixtures[row]['ids'])and sum(not c['prefill']for c in calls)==len(ids)-1
   sig.append((r['prompt_IDs_SHA256'],r['generated_IDs_SHA256'],len(ids),r['natural_EOS']))
 if signature is None:signature=sig
 assert sig==signature
 owners.append({'stage':q['stage'],'rank':q['rank'],'path':str(p),'SHA256':rec['SHA256']})
 if(q['stage'],q['rank'])==(3,0):tail=q
assert len(owners)==32 and {(o['stage'],o['rank'])for o in owners}=={(s,r)for s in range(4)for r in range(8)}
clientdir=R/'results'/('GLM-r355-client-'+run);client=json.loads((clientdir/'result.json').read_bytes());assert client['completed']and client['passed']and client['HTTP_requests']==12
assert sha(clientdir/'result.json')==w['evidence']['client_SHA256']
for pair in client['pairs']:
 for row,req in enumerate(pair['requests']):
  p=clientdir/('pair%d-row%d.wire'%(pair['index'],row));assert sha(p)==req['wire_SHA256'];wire=json.loads(p.read_bytes());usage=validate_reply(wire,fixtures[row]['fixture'],fixtures[row]['gold']);assert usage['completion_tokens']==tail['pairs'][pair['index']]['requests'][row]['tokens'] and req['quality_passed']
rows=[]
for i in range(6):
 aligned=[(json.loads(Path(o['path']).read_bytes())['pairs'][i]['complete_schedule_wall_s'],o['stage'],o['rank'])for o in owners];users=[]
 for req in tail['pairs'][i]['requests']:
  ticks=req['emitter_token_monotonic'];ids=req['generated_ids'];assert len(ticks)==len(ids)and all(math.isfinite(x)for x in ticks)and ticks==sorted(ticks)
  visible=[t for t,token in zip(ticks,ids)if token not in[154820,154827,154829]];admission=req['emitter_admission_monotonic'];assert admission<=visible[0]<visible[-1];gaps=[b-a for a,b in zip(visible,visible[1:])]
  users.append({'row':req['row'],'useful_tokens':len(visible),'TTFT_s':visible[0]-admission,'first_to_last_decode_tok_s':(len(visible)-1)/(visible[-1]-visible[0]),'max_ITL_s':max(gaps),'admission':admission,'last_visible':visible[-1]})
 interval=max(u['last_visible']for u in users)-min(u['admission']for u in users)
 rows.append({'index':i,'warmup':i<2,'enabled':PLAN[i],'max32_complete_schedule_wall_s':max(aligned)[0],'argmax':max(aligned)[1:],'users':users,'emitter_total_tok_s':sum(u['useful_tokens']for u in users)/interval,'emitter_per32card_tok_s':sum(u['useful_tokens']for u in users)/interval/32})
a=statistics.mean(rows[i]['max32_complete_schedule_wall_s']for i in [2,5]);b=statistics.mean(rows[i]['max32_complete_schedule_wall_s']for i in [3,4])
final=R/'results'/('GLM-r355-final-audit-'+run+'.json');f=json.loads(final.read_bytes());assert f['all_independent_post_passed']and all(p['passed']and not p['expected_PIDs_still_exist']and not p['matching_owner_or_client_PIDs']and p['KFD_empty']and p['port43843_bindable']for p in f['independent_post'])
account=subprocess.check_output(['sacct','-S','2026-11-01','-j','910407.689,910371.649,910371.650,910374.1265,910374.1266','-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed'],text=True,timeout=25)
assert len(account.splitlines())==5 and all('|COMPLETED|0:0|'in s for s in account.splitlines())
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'root_actual_source_wire_owner_timeline_audit':True,'owner_count':32,'HTTP_gold':12,'owners':owners,'rows':rows,'OFF_mean_s':a,'ON_mean_s':b,'rate_gain_fraction':a/b-1,'latency_reduction_fraction':1-b/a,'OFF_endpoint_drift':rows[5]['max32_complete_schedule_wall_s']/rows[2]['max32_complete_schedule_wall_s']-1,'Slurm':account,'independent_post_evidence_SHA256':sha(final),'watch_SHA256':sha(wpath),'minimum_observed_driver_free_bytes':f['minimum_observed_driver_free_bytes_all32'],'scope':'One fixed two-request staticC2 Modelpair,not online/client incremental/benchmark/15 qualification','raw_payload_exported':False}))
