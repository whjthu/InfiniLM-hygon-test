"""r347 final artifact visibility/identity, specialized to two ordinary requests."""
from pathlib import Path
import hashlib,json,time
from contract_r410 import RUNTIME_SHA,NATIVE_SHA,TARGET_NODES,PROMPT_SHA
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token_metrics(request):
 events=request['events'];assert[t['token']for t in events]==request['generated_ids']
 useful=[v for v in events if v['token']not in[163585,163586]];times=[v['at_s']for v in useful];assert all(b>=a for a,b in zip(times,times[1:]))
 gaps=sorted(b-a for a,b in zip(times,times[1:]))
 def pct(q):
  if not gaps:return None
  k=(len(gaps)-1)*q;i=int(k);j=min(i+1,len(gaps)-1);return gaps[i]+(gaps[j]-gaps[i])*(k-i)
 rate=(len(times)-1)/(times[-1]-times[0])if len(times)>1 and times[-1]>times[0]else None
 return dict(domain='Each node0 request-relative monotonic after before-reset and after full4leader token gather; engine driver, not HTTP/client',generated_including_EOS=len(events),visible_non_EOS=len(useful),EOS_count=len(events)-len(useful),TTFT_s=times[0]if times else None,first_last_visible_s=times[-1]-times[0]if times else None,visible_first_last_tok_s=rate,visible_first_last_tok_per_service32card_s=rate/32 if rate is not None else None,ITL_P95_s=pct(.95),ITL_P99_s=pct(.99),ITL_max_s=max(gaps)if gaps else None,request_generation_through_EOS_s=request['full_request_s'],parse_and_after_reset_s=None,parse_and_after_reset_timing='UNKNOWN; normal reset/quality are checked but frozen runtime has no exclusive clocks for them')
def terminal_report(path,until,clock=time.monotonic,sleep=time.sleep):
 while True:
  try:
   q=json.loads(Path(path).read_bytes())
   if q.get('completed')is True:return q
   # Explicit final failure must not be retried/hidden as visibility loss.
   if q.get('phase')=='failed'and'cleanup_errors'in q:return q
  except(FileNotFoundError,json.JSONDecodeError):pass
  if clock()>=until:raise TimeoutError('Original completed step report visibility unresolved; no replay')
  sleep(min(1,max(0,until-clock())))
def collect(root,run,actual,until):
 rows=[];signatures=[];timing=None
 for node in range(4):
  path=Path(root)/'results'/('Kimi-ordinary-r410-'+run+'-node'+str(node)+'.json');q=terminal_report(path,until)
  assert q['completed']and q['all8_owned_workers_shutdown_completed']and q['sampler_CPU_cleanup_completed']and q['control_CPU_cleanup_completed']and not q['cleanup_errors']
  assert q['actual_step']==actual and q['node']==node and q['source_manifest_SHA256']==RUNTIME_SHA and q['Native_SHA256']==NATIVE_SHA
  runtime=Path(root)/'incoming/Kimi-TP8PP4-ordinary-r410/runtime-v2'
  specs=json.loads((runtime/'CAPI_PROVIDERS.json').read_bytes())['symbols'];providers=q['actual_CAPI_providers'];maps=q['actual_loaded_libraries']
  assert set(providers)==set(specs)and len(providers)==20
  for name,spec in specs.items():
   assert providers[name]==spec['path'],dict(symbol=name,actual=providers[name],expected=spec['path'])
   actual_digest=q['Native_SHA256']if spec['path'].endswith('/libKimi_TP8PP4_ordinary_r410.so')else maps['libinfinilm_runtime.so']['SHA256']
   assert actual_digest==spec['SHA256']
  assert q['layout']['TP']==8 and q['layout']['PP']==4 and q['layout']['PP_stage']==node and q['layout']['sampling_node']==3
  assert q['all93_stage_keys']['all93_layer_ids_exactly_once_across_PPstages']and q['owner_affinity']['verified']
  pair=q['ordinary_report'];assert pair['completed']and pair['draft_cards']==0 and len(pair['requests'])==2
  signature=[]
  assert q['last_memory_guard']['minimum_free_bytes']>4<<30
  for r,name,length,prompt_hash in zip(pair['requests'],['retained8635','near32k'],[8635,32638],PROMPT_SHA):
   assert r['name']==name and r['input_tokens']==length and r['output_budget']==128 and r['completed']and r['normal_reset']and r['finish']=='eos'and r['quality']['passed']
   assert 1<=len(r['generated_ids'])<=128 and r['generated_ids'][-1]in[163585,163586]
   assert r['prompt_ID_SHA256']==prompt_hash and hashlib.sha256(json.dumps(r['generated_ids'],separators=(',',':')).encode()).hexdigest()==r['output_ID_SHA256']
   signature.append((name,r['prompt_ID_SHA256'],r['output_ID_SHA256'],len(r['generated_ids']),r['finish']))
  count=sum((r['input_tokens']+511)//512+len(r['generated_ids'])-1 for r in pair['requests'])
  assert len(q['ordinary_calls'])==count and q['output_metrics']['engine_calls']==count
  assert [v['serial']for v in q['ordinary_calls']]==list(range(count))and all(v['validated']and v['CAPI_return']==0 for v in q['ordinary_calls'])
  if node==0:timing=[token_metrics(v)for v in pair['requests']]
  signatures.append(signature);rows.append(dict(node=node,physical_node=TARGET_NODES[node],path=str(path),SHA256=sha(path),normal8owners=True,prefactory_20_exact_providers_saved=True))
 assert all(s==signatures[0]for s in signatures)
 return dict(target_reports=rows,all32_normal=True,four_leader_request_signatures_equal=True,request_signatures=signatures,emitter_timing=timing,performance_requires_root_audit=True,HTTP=False,client_SLO_qualified=False,topology_package_not_communication_only=True,fresh_TP16_control=False,historical_diagnostic_comparison_only=True)
