"""Independent terminal361 audit, actual ticks and full pair costs; metadata only."""
from pathlib import Path
import json,hashlib,sys,math,statistics,subprocess,datetime,traceback
R=Path.home()/'hygon-inference-refactor-20261014';run='GLM-C2M1-r361-a';D=R/'incoming/GLM-prefill-window-interleave-r361/runtime-r1';O=D/'owner';C=D/'controller'
PIN='5c85fe82960f85c4a2e601fb0e58442c3ae11e2797a4dcec6d957c61af9b3626';CPIN='1fdca2d847002dc94871470491945cd97dd660b6352eb27471cdc828e81d8205';NATIVE='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_bytes())
q=dict(completed=False,passed=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,source_commit='48f2b4df97e764df5d5708d355555923b3f83bb1',owner_manifest_SHA256=PIN,controller_manifest_SHA256=CPIN,scope='Static C2M1 twelve HTTP/six pairs, fixeddecode-deferON; only W4-boundary prefillinterleave changes',onlineSLO_or_15_or_broadbenchmark_qualified=False,raw_body_wire_and_full_IDs_remain_private=True)
def pct(values,p):
 v=sorted(values);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def metric(pair):
 rows=[];all_ticks=[];admissions=[]
 for row in pair['requests']:
  ids=row['generated_ids'];ticks=row['emitter_token_monotonic'];admission=row['emitter_admission_monotonic']
  assert len(ids)==len(ticks)and all(math.isfinite(t)for t in ticks)and ticks==sorted(ticks)
  v=[t for token,t in zip(ids,ticks)if token not in[154820,154827,154829]];assert len(v)>1 and admission<=v[0]
  gaps=[b-a for a,b in zip(v,v[1:])];duration=v[-1]-v[0];assert duration>0
  rows.append(dict(row=row['row'],fixture_id=row['fixture_id'],visible_tokens=len(v),TTFT_s=v[0]-admission,first_to_last_s=duration,user_decode_tok_s=(len(v)-1)/duration,ITL_max_s=max(gaps),ITL_P95_s=pct(gaps,.95),ITL_P99_s=pct(gaps,.99),full_timeline=True))
  all_ticks+=v;admissions.append(admission)
 interval=max(all_ticks)-min(admissions);visible=sum(r['visible_tokens']for r in rows)
 good=sum(r['visible_tokens']for r in rows if r['user_decode_tok_s']>15)
 return dict(requests=rows,useful_output_tokens=visible,admission_to_last_visible_s=interval,total_tok_s=visible/interval,total_tok_s_per_all32cards=visible/interval/32,decode_gt15_qualified_requests=sum(r['user_decode_tok_s']>15 for r in rows),decode_gt15_goodput_tok_s_per_all32cards=good/interval/32,clock='Same emitter monotonic; actual admissions and emissions including long prefill waits; HTTP is buffered')
try:
 sys.path.insert(0,str(C));import controller
 go=load(R/'results/GLM-r361-execution-GO-48f2b4df.json');controller.verify(R,C,go)
 assert sha(O/'SOURCE_MANIFEST.json')==PIN and sha(C/'SOURCE_MANIFEST.json')==CPIN
 watch=R/'results'/('GLM-r361-watch-'+run+'.json');w=load(watch);assert w['completed']and w['passed']and w['post_empty']
 actual={int(k):v for k,v in w['actual_stages'].items()};assert len(actual)==4
 assert actual=={0:'910371.667',1:'910371.668',2:'910374.1286',3:'910374.1287'}
 assert all(o['srun_exit']==0 and o['row'][0]==actual[int(s)]and o['row'][2:4]==['COMPLETED','0:0']for s,o in w['observations'].items())
 accounting=subprocess.check_output(['sacct','-S','2026-11-01','-j',','.join([*actual.values(),'910407.716']),'-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
 account_rows={r[0]:r for r in (line.split('|')for line in accounting.splitlines()if line.strip())}
 assert all(account_rows[step][1:3]==['COMPLETED','0:0']for step in [*actual.values(),'910407.716'])
 class View:
  def __init__(self):self.actual=actual
  def label(self,s):return 'GLM-r361-'+run+'-s%d'%s
 evidence=controller.collect(R,View(),go)
 card=load(O/'FIXTURE_CARD.json');fixtures=card['fixtures'];assert[len_['current_prepared_prompt_tokens']for len_ in fixtures]==[479,8900]
 owners=[];tail=None;signatures=[]
 for stage in range(4):
  for rank in range(8):
   path=R/'results'/('GLM-r361-'+run+'-stage%d-rank%d.json'%(stage,rank));a=load(path)
   assert a['actual_producer']==actual[stage]and a['source_manifest_SHA256']==PIN and a['library_sha256']==NATIVE and a['observer_native_destroy_returned0']
   assert a['cache_after_destroy']['active']==a['cache_after_destroy']['active_bytes']==0 and a['cache_after_destroy']['created']==a['cache_after_destroy']['destroyed']
   sig=[]
   for i,pair in enumerate(a['pairs']):
    assert pair['decode_defer_fixed_both_arms']and pair['prefill_consumption']['qualified']
    for row,req in enumerate(pair['requests']):
     f=fixtures[row];ids=req['generated_ids'];assert req['prompt_IDs_SHA256']==f['current_prepared_prompt_IDs_SHA256']and req['tokens']==len(ids)and ids[-1]in[154820,154827,154829]
     assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==req['generated_IDs_SHA256']
     calls=[c for c in pair['calls']if c['row']==row];pf=[c for c in calls if c['prefill']];dec=[c for c in calls if not c['prefill']];n=f['current_prepared_prompt_tokens']
     assert[(c['past'],c['T'])for c in pf]==[(p,min(512,n-p))for p in range(0,n,512)]
     assert[(c['past'],c['T'])for c in dec]==[(p,1)for p in range(n,n+len(ids)-1)]
     assert all(math.isfinite(c['Native_CAPI_s'])and c['Native_CAPI_s']>=0 for c in calls)
     sig.append((i,row,req['prompt_IDs_SHA256'],req['generated_IDs_SHA256'],req['tokens']))
    events=pair['prefill_window_interleave_events'];groups=[e['forwards']for e in events if e['kind']=='complete_long_prefill_group'];adv=[e for e in events if e['kind']=='short_decode_at_complete_long_W4_boundary']
    assert groups==([4,4,4,4,2]if pair['enabled']else[])
    assert len(adv)==(5 if pair['enabled']and pair['requests'][0]['tokens']>6 else pair['prefill_consumption']['short_advances'])
   signatures.append(sig);owners.append(dict(stage=stage,rank=rank,path=str(path),SHA256=sha(path),normal=True,minimum_driver_free_bytes=a['minimum_driver_free_bytes']))
   if(stage,rank)==(3,0):tail=a
 assert all(s==signatures[0]for s in signatures)
 clientdir=R/'results'/('GLM-r361-client-'+run);client=load(clientdir/'result.json');assert client['completed']and client['passed']and client['attempted_HTTP']==client['completed_HTTP']==client['quality_passed_HTTP']==12
 sys.path.insert(0,str(O));from workload_r349 import validate_reply
 seen=set();journal={r['request_id']:r for r in tail['queue_journal']['requests']};assert len(journal)==12
 for i,pair in enumerate(client['pairs']):
  for row,c in enumerate(pair['requests']):
   f=fixtures[row];source=Path(f['request_external_path'].replace('/refactor',str(R),1));assert sha(source)==f['request_SHA256'];body=load(source)
   if row==0:gold={'path':'input.json'}
   else:
    data=json.loads(next(m['content']for m in body['messages']if m['role']=='tool'and'records'in m['content']));values={x['id']:x['value']for x in data['records']};gold={'a':values[data['query_ids'][0]],'b':values[data['query_ids'][1]]}
   p=clientdir/('pair%d-row%d.wire'%(i,row));assert sha(p)==c['wire_SHA256'];wire=load(p);usage=validate_reply(wire,f,gold);r=tail['pairs'][i]['requests'][row]
   assert wire['id']==c['request_id']and wire['id']not in seen;seen.add(wire['id']);assert usage['completion_tokens']==r['tokens']and usage['prompt_tokens']==r['prompt_tokens']and c['quality_passed']
   j=journal[wire['id']];assert j['finished']and j['original_reset_all32_completed']and j['native_forward_calls']==math.ceil(r['prompt_tokens']/512)+r['tokens']-1
 aligned=[];emitter=[]
 for i in range(6):
  durations=[]
  for owner in owners:
   a=load(owner['path']);durations.append((a['pairs'][i]['complete_schedule_wall_s'],owner['stage'],owner['rank']))
  maximum=max(durations);aligned.append(dict(index=i,enabled=i in[1,3,4],warmup=i<2,max32_complete_schedule_wall_s=maximum[0],argmax_stage=maximum[1],argmax_rank=maximum[2]));emitter.append(dict(index=i,**metric(tail['pairs'][i])))
 off=statistics.mean(aligned[i]['max32_complete_schedule_wall_s']for i in[2,5]);on=statistics.mean(aligned[i]['max32_complete_schedule_wall_s']for i in[3,4]);primary={}
 for row in[0,1]:
  primary[str(row)]={name:{'OFF':[emitter[i]['requests'][row][name]for i in[2,5]],'ON':[emitter[i]['requests'][row][name]for i in[3,4]],'OFF_mean':statistics.mean(emitter[i]['requests'][row][name]for i in[2,5]),'ON_mean':statistics.mean(emitter[i]['requests'][row][name]for i in[3,4])}for name in['TTFT_s','first_to_last_s','user_decode_tok_s','ITL_max_s','ITL_P95_s','ITL_P99_s']}
 HTTP=[]
 for pair in client['pairs']:
  rows=pair['requests'];assert all(r['common_CPU_HTTP_ended']>=r['common_CPU_HTTP_started']for r in rows)
  HTTP.append(dict(index=pair['index'],enabled=pair['enabled'],warmup=pair['warmup'],rows=[dict(row=r['row'],HTTP_wall_s=r['HTTP_wall_s'],request_id=r['request_id'],wire_SHA256=r['wire_SHA256'])for r in rows],complete_pair_wall_s=max(r['common_CPU_HTTP_ended']for r in rows)-min(r['common_CPU_HTTP_started']for r in rows),clock='Single client process monotonic; complete buffered HTTP wall, not HTTP token delivery'))
 q.update(HTTP_trials=HTTP,all32_minimum_observed_driver_free_bytes=min(o['minimum_driver_free_bytes']for o in owners),memory_is_peak=False)
 post_code='''from pathlib import Path
import os,json,socket,sys
expected=json.loads(sys.argv[1]);pattern=b'/GLM-prefill-window-interleave-r361/runtime-r1/owner/'
k=Path('/sys/class/kfd/kfd/proc');assert k.is_dir();kfd=sorted(int(p.name)for p in k.iterdir()if p.name.isdigit());owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes()
 except(OSError,PermissionError):continue
 if pattern in cmd and(b'model_entry_r290.py'in cmd or b'client_r355.py'in cmd):owned.append(int(p.name))
alive=[pid for pid in expected if Path('/proc',str(pid)).exists()]
with socket.socket()as s:
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 try:s.bind(('127.0.0.1',43843));free=True
 except OSError:free=False
print(json.dumps(dict(CPU_only=True,GPU_initialized=False,node=socket.gethostname().split('.')[0],actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],expected_owner_PIDs=expected,expected_PIDs_still_exist=alive,matching_owner_or_client_PIDs=owned,KFD_pids=kfd,port43843_bindable=free,passed=not alive and not owned and not kfd and free)))
'''
 independent_post=[]
 for stage,(parent,node)in enumerate([(910371,'a02r3n03'),(910371,'a02r3n07'),(910374,'a02r4n04'),(910374,'a02r4n05')]):
  phasefiles=[R/'results'/('GLM-Model-phase-r9183-'+run+'-stage%d-rank%d.jsonl'%(stage,rank))for rank in range(8)]
  expected=[load_p['PID']for p in phasefiles for load_p in [json.loads(p.read_bytes().splitlines()[-1])]];assert len(set(expected))==8
  args=['srun','--jobid='+str(parent),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-r361-independent-post-s%d'%stage,'python3','-B','-c',post_code,json.dumps(expected)]
  p=subprocess.run(args,text=True,capture_output=True,timeout=150);assert p.returncode==0,p.stderr[-1000:]
  rows=[json.loads(line)for line in p.stdout.splitlines()if line.startswith('{')];assert len(rows)==1 and rows[0]['node']==node and rows[0]['passed'];independent_post.append(rows[0])
 q.update(passed=True,actual_stages=actual,keeper='910407.716',accounting=accounting,all4_srun_exit0=True,all5_original_handles_COMPLETED0=True,all12_independent_wire_gold=True,all32_normal_and_signatures_equal=True,owners=owners,controller_evidence=evidence,watch_SHA256=sha(watch),client_SHA256=sha(clientdir/'result.json'),aligned_trials=aligned,emitter=emitter,primary_full_request_metrics=primary,complete_pair_OFF_mean_s=off,complete_pair_ON_mean_s=on,complete_pair_rate_gain_fraction=off/on-1,OFF_endpoint_drift_fraction=abs(aligned[5]['max32_complete_schedule_wall_s']/aligned[2]['max32_complete_schedule_wall_s']-1),ON_endpoint_drift_fraction=abs(aligned[4]['max32_complete_schedule_wall_s']/aligned[3]['max32_complete_schedule_wall_s']-1),fresh_before=w['fresh_before'],post=w['post'],independent_post=independent_post,all_independent_post_passed=True,no_cross_host_clock_subtraction_or_overlapping_stage_sums=True,remaining_decode_window_not_used_as_total_gain=True,promotion='Root review required: report maxITL benefit with P99/longTTFT/fullpair/goodput costs, never trim longprefill wait or claim15/onlineSLO')
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
finally:q['completed']=True;print(json.dumps(q))
