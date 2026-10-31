"""Whole-entrypoint CPU witness. Synthetic HTTP at urllib boundary; no quality claim."""
import argparse,copy,datetime,hashlib,importlib.util,json,os,pathlib,signal,socket,subprocess,sys,time,traceback,urllib.request
from types import SimpleNamespace
signal.alarm(205)
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');S=R/'incoming/Kimi-latest-service-r275';O=R/'candidates/Kimi-supplement-client-r278';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10))
def module(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sys.path.insert(0,str(D));plan=json.loads((D/'PLAN.json').read_bytes())
p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert sha(D/'SOURCE_MANIFEST.json')==a.manifest_sha
cm=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in cm['files_SHA256'].items())
assert sha(S/plan['service_manifest_name'])==plan['service_manifest_SHA256'];sm=json.loads((S/plan['service_manifest_name']).read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
model=json.loads((S/'runtime/model-manifest-r9183.json').read_bytes());assert model['request_prefill_modes_r252']==['W512']*41 and model['profile_admitted_indices_r262']==[39]
old=R/'incoming/Kimi-latest-service-r267';om=json.loads((old/'service-source-manifest-r267.json').read_bytes());changed=[n for n,h in sm['files_SHA256'].items()if h!=om['files_SHA256'][n]];assert sorted(changed)==['runtime/model-manifest-r9183.json','runtime/target_inside.sh','runtime/target_model.py','runtime/target_protocol.py']
# Source equivalence after exactly undoing namespace-only edits.
for n in changed:
 if n.endswith('.json'):continue
 assert (S/n).read_text().replace('Kimi-latest-service-r275','Kimi-latest-service-r267').replace('Kimi-latest-service-stop-r275','Kimi-latest-service-stop-r267').replace('Kimi-latest-service-identity-r275','Kimi-latest-service-identity-r267').replace('Kimi-latest-service-ready-r275','Kimi-latest-service-ready-r267')==(old/n).read_text()
result={'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'production_HTTP':0,'synthetic_HTTP_only':True,'quality_or_performance_claim':False,'source_manifest_SHA256':a.manifest_sha,'service_manifest_SHA256':plan['service_manifest_SHA256'],'changed_service_members':changed,'real_transport_loader_validators_and_client_mains':True,'cases':[]}
def no_network(*a,**k):raise RuntimeError('CPU preflight forbids network')
socket.create_connection=no_network;socket.socket.connect=no_network
realrun=subprocess.run;realurl=urllib.request.urlopen
try:
 for scenario in ['positive','stale-transport','wrong-manifest','wrong-profile-fixture','wrong-admitted-index']:
  root=O/scenario;root.mkdir();rr=root/'runtime';(rr/'results').mkdir(parents=True);source=S
  if scenario in ['stale-transport','wrong-manifest']:
   import shutil
   source=root/'service';shutil.copytree(S,source)
   bad=source/('client/client_p0_r143.py'if scenario=='stale-transport'else plan['service_manifest_name']);bad.write_bytes(bad.read_bytes()+b'\n')
  run='CPU_SYNTHETIC_'+scenario;os.environ['GLOBAL_RUN_R2354']=run;os.environ['KIMI_R205_CLIENT_DEADLINE_UNIX']=str(time.time()+8000)
  ident=rr/'identity.json';dump(ident,{'native_SHA256':sm['Native_SHA256'],'manifest_SHA256':plan['model_manifest_SHA256'],'protocol_commit':sm['upstream_commit'],'actual_step':'CPU_SYNTHETIC_NOT_GPU','cards':32,'TP':16,'PP':2,'max_context_tokens':32768})
  reportpath=rr/'results'/('Kimi-latest-service-r275-'+run+'-node0.json');q={'run':run,'node':0,'manifest_SHA256':plan['model_manifest_SHA256'],'agent_requests':[],'sparse_model_profiles':[]}
  ready=rr/'results'/('Kimi-latest-service-ready-r275-'+run+'.json');dump(ready,{'ready':True,'actual_step':'CPU_SYNTHETIC_NOT_GPU','identity_path':str(ident),'URL_on_service_node':'http://cpu.invalid/v1'})
  fixed=rr/'candidates/Kimi-latest-service-r267/fixed-real-request-r267.json';fixed.parent.mkdir(parents=True);original=R/'candidates/Kimi-latest-service-r267/fixed-real-request-r267.json';fixed.write_bytes(original.read_bytes());fixture=json.loads(fixed.read_bytes());assert sha(fixed)=='9c488dcb625b72e9dfb3d0b001ec95878b4d0d1366785c1acb84e44f1ad0b2f9'
  if scenario=='wrong-profile-fixture':fixed.write_bytes(fixed.read_bytes()+b'\n')
  calls=[];multi_started=[];transport_modules=[]
  def publish():
   for node in range(4):
    z=copy.deepcopy(q);z['node']=node;dump(rr/'results'/('Kimi-latest-service-r275-'+run+'-node%d.json'%node),z)
  publish()
  class Response:
   status=200;headers={'content-type':'application/json'}
   def __init__(self,raw):self.raw=raw
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def read(self):return self.raw
  def synthetic_urlopen(req,timeout=None):
   body=json.loads(req.data);idx=len(calls);assert idx<41 and timeout>0;calls.append(copy.deepcopy(body));messages=body['messages'];thinking=body['chat_template_kwargs']['enable_thinking'];budget=body['max_tokens'];session=body['user'];assert budget==(8192 if thinking else 4096);assert body['stream']is False
   if idx<4:
    assert body['parallel_tool_calls']is True and body['tools'][0]['function']['name']=='read_file'
    if idx%2==0:
     assert body['tool_choice']=='required';message={'role':'assistant','content':'','tool_calls':[{'id':'opaque-'+str(idx)+'-'+str(i),'type':'function','function':{'name':'read_file','arguments':json.dumps({'path':p})}}for i,p in enumerate(['left.json','right.json'])]};finish='tool_calls'
    else:
     assert body['tool_choice']=='none';tool=[v for v in messages if v['role']=='tool'];original_calls=next(v['tool_calls']for v in messages if v.get('tool_calls'));assert [v['tool_call_id']for v in tool]==list(reversed([v['id']for v in original_calls]));byid={v['tool_call_id']:json.loads(v['content'])for v in tool};expected={}
     for c in original_calls:
      name=json.loads(c['function']['arguments'])['path'];data=byid[c['id']];vals={v['id']:v['value']for v in data['records']};expected[name]={'nonce':data['nonce'],'sum':sum(vals[k]for k in data['query_ids'])}
     expected['total']=sum(v['sum']for v in expected.values());message={'role':'assistant','content':json.dumps(expected)};finish='stop'
    prompt=9000
   elif idx<36:
    assert body['tool_choice']=='none'and not thinking;case=json.loads((S/'benchmark/gsm8k-fixed-random32.json').read_bytes())['cases'][idx-4];assert messages[-1]['content']==case['question'];message={'role':'assistant','content':'#### '+case['gold_numeric']};finish='stop';prompt=900
   else:
    assert messages==fixture['body']['messages'] and body['tool_choice']==fixture['body']['tool_choice'];message={'role':'assistant','content':'','tool_calls':[{'id':'opaque-profile','type':'function','function':{'name':'calculate','arguments':json.dumps(fixture['expected_calculate_arguments'])}}]};finish='tool_calls';prompt=fixture['expected_prompt_tokens']
   if thinking:message['reasoning_content']='SYNTHETIC_CPU_REASONING'
   assert prompt+budget<=32768
   rid='CPU-SYNTHETIC-%02d'%idx;rc=[hashlib.sha256((v.get('reasoning_content')or v.get('reasoning')).encode()).hexdigest()for v in messages if v.get('role')=='assistant'and(v.get('reasoning_content')or v.get('reasoning'))];promptsha=fixture['expected_Native_prompt_IDs_SHA256']if idx>=36 else 'CPU_SYNTHETIC_PROMPT_IDS'
   audit={'qualified':True,'prequeue_bound':True,'markers_sent_to_Native':False,'prompt_tokens':prompt,'official_reencode_equals_actual_prompt_and_IDs':True,'generation_prefix_correct':True,'generation_thinking':thinking,'source_RC_SHA256_sequence':rc,'RC_one_contribution_per_source_field':True,'RC_order_preserved':True,'actual_prompt_SHA256':'CPU_SYNTHETIC_PROMPT','Native_prompt_token_ids_SHA256':promptsha}
   row={'request_id':rid,'admitted_index':idx+(1 if scenario=='wrong-admitted-index'and idx==0 else 0),'client_session':session,'prefill_mode':'W512','input_tokens':prompt,'output_budget':budget,'normal_reset':True,'finish':'eos','generated_ids':[7,8,163585],'history_render_audit':audit,'long_prefill_progress':{'input_tokens_committed':prompt},'long_prefill_observations':[{'full_prompt_consumed':True,'input_tokens_committed':prompt,'prefill_wall_s':1.,'recorded_completed_forward_CAPI_maxhost_s_sum':.8}],'started_monotonic':100.+idx*5,'finished_monotonic':103.+idx*5,'backend_token_monotonic':[101.+idx*5,102.+idx*5,103.+idx*5]};q['agent_requests'].append(row)
   rec={'label':'agent-'+rid,'verify_host_samples':[{'verify_index_zero_based':2,'CAPI_maxhost_s':.05}]}
   if idx==39:rec.update(armed=True,actual_forward_map=[{'C':1,'T':1,'widths':[1],'actual_verify_index_zero_based':2}],raw_decoder_owner_events={'all8_published':True,'owners':{str(i):{'SYNTHETIC':True}for i in range(8)}},host_adapter_s=.001,decoder_summary={'SYNTHETIC_CPU_ONLY':True})
   q['sparse_model_profiles'].append(rec);publish();usage={'prompt_tokens':prompt,'completion_tokens':3,'total_tokens':prompt+3};return Response(json.dumps({'id':rid,'choices':[{'message':message,'finish_reason':finish}],'usage':usage}).encode())
  urllib.request.urlopen=synthetic_urlopen
  def nested_run(cmd,**kwargs):
   assert len(cmd)>3 and pathlib.Path(cmd[2]).resolve()==(D/'client_multicall_r278.py').resolve();multi_started.append(True);m=module(D/'client_multicall_r278.py','cpu_real_multicall_'+scenario);argv=sys.argv;sys.argv=[str(D/'client_multicall_r278.py')]+cmd[3:]
   try:rc=m.main()
   finally:sys.argv=argv
   return SimpleNamespace(returncode=rc)
  subprocess.run=nested_run;m=module(D/'client_r278.py','cpu_real_controller_'+scenario);m.R=rr;m.S=source
  argv=sys.argv;sys.argv=[str(D/'client_r278.py'),'--base-url','http://cpu.invalid/v1','--model','Kimi-K3','--output-dir',str(root/'client'),'--identity',str(ident),'--service-report',str(reportpath),'--timeout','3600']
  try:ret=m.main()
  finally:sys.argv=argv
  report=json.loads((root/'client/result.json').read_bytes());assert report['completed'];mc=json.loads((root/'client/multicall/result.json').read_bytes());case={'name':scenario,'main_returncode':ret,'main_passed':report['passed'],'real_multicall_entry_executed':bool(multi_started),'synthetic_HTTP':len(calls),'multicall_passed':mc['passed'],'multicall_error_class':mc.get('error_class'),'multicall_error_SHA256':mc.get('error_SHA256'),'controller_error':report.get('error'),'result_SHA256':sha(root/'client/result.json')}
  print(json.dumps(dict(case,observed_before_validation=True)),flush=True)
  if scenario=='positive':
   assert ret==0 and report['passed']and len(calls)==41 and report['actual_HTTP_requests']==41 and report['actual_Native_requests']==41 and report['normal34_repeated']is False;assert report['C1_profile']['attribution_qualified'];assert report['GSM8K']['correct']==32;assert [v['admitted_index']for v in q['agent_requests']]==list(range(41));case.update(synthetic_GSM32_correct=32,synthetic_profile_all32_covered=True,budgets_full=True)
  elif scenario in ['stale-transport','wrong-manifest']:assert ret==1 and not report['passed'] and len(calls)==0 and not mc['passed']
  elif scenario=='wrong-profile-fixture':assert ret==1 and not report['passed'] and len(calls)==36 and mc['passed'] and 'AssertionError'in report['error']
  else:assert ret==1 and not report['passed'] and len(calls)==2 and not mc['passed']
  case['passed']=True;result['cases'].append(case);print(json.dumps(case),flush=True)
 result['passed']=len(result['cases'])==5
except BaseException as e:result.update(error=repr(e),failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(e.__traceback__)])
finally:
 subprocess.run=realrun;urllib.request.urlopen=realurl;result['completed']=True;dump(O/'CPU-prepared.json',result);print(json.dumps({k:result.get(k)for k in ['completed','passed','actual_step','error','failure_locations']}),flush=True)
assert result['passed']
