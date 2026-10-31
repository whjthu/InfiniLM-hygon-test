"""Actual Deep client mains and transport. Synthetic responses only at urllib boundary."""
import pathlib,json,hashlib,sys,os,signal,shutil,datetime,time,importlib.util,urllib.request,socket,copy,ast,traceback
from types import SimpleNamespace
signal.alarm(205)
R=pathlib.Path('/refactor');D=R/'incoming/Deep-client-entry-r286';S=R/'incoming/Deep-latest-full-owner-r285';O=R/'candidates/Deep-client-entry-r286';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10))
def module(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'production_HTTP':0,'synthetic_owner_and_quality_only':True,'real_multicall_main_quality_main_transport_and_validators':True,'unchanged_normal35_not_rerun':True,'cases':[]}
realurl=urllib.request.urlopen
try:
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert sha(D/'SOURCE_MANIFEST.json')==a.manifest_sha;mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in mf['files_SHA256'].items());expected=mf['owner_source_manifest_SHA256'];assert sha(S/'SOURCE_MANIFEST.json')==expected;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items());q['owner_source_manifest_SHA256']=expected
 for n in sm['files_SHA256']:
  if n.endswith('.py'):ast.parse((S/n).read_bytes(),filename=str(S/n))
 # The four formerly checked loader/guard components remain immutable apart from intended edits.
 sys.path.insert(0,str(S));os.environ['DEEP_SOURCE_MANIFEST_SHA']=expected
 # Exercise actual preweight module imports without invoking its GPU function.
 import importlib.abc
 class NoGPUImport(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0]in ['torch','infinilm','_infinilm']:raise RuntimeError('CPU preweight dependency import forbids '+fullname)
 guard=NoGPUImport();sys.meta_path.insert(0,guard)
 import preweight_r206b as preweight
 assert 'profiles_r206b'not in sys.modules,'Legacy helper must be deferred until source verification'
 import profiles_r206b as legacyprofiles
 import chat_prepare_r173 as legacychat
 assert pathlib.Path(legacychat.encoder.__file__).resolve()==pathlib.Path('/refactor/incoming/Deep-agent-regular32K-r206b/encoding/encoding_dsv4.py').resolve()
 assert sha(legacychat.encoder.__file__)=='838447d0e81fe8e106899aa56b2584607c4c3e63202a4ec8d6d333248866f798'
 assert 'torch'not in sys.modules and 'infinilm'not in sys.modules and '_infinilm'not in sys.modules
 q['actual_preweight_and_legacy_dependency_import_guarded_PASS']=True;q['legacy_encoder_used_for_actual_requests']=False;q['private_encoder_body_copied']=False
 from native_binding_r206b import binding
 b,receipt,normalized=binding();assert len(normalized['link']['full']['objects'])==48
 newext='/refactor/candidates/latest-family-native-r269/Deep/python/_infinilm.cpython-310-x86_64-linux-gnu.so';assert normalized['actual_DTK_bindings'][newext]['resolved']==str(pathlib.Path(newext).resolve());q['resolved_metadata_correct']=True
 import bootstrap_r206b as bootstrap
 tree,proof=bootstrap.transform(bootstrap.BASE.read_bytes());assert proof['Native_forward_AST_unchanged']and proof['original_finally_destroy_retained'];q['loader_AST']=proof
 plan=json.loads((S/'PREPARATION.json').read_bytes());assert plan['HTTP']==103 and plan['Native']==102 and plan['ordinary_output_budget']==4096 and plan['thinking_output_budget']==8192
 def no_network(*a,**k):raise RuntimeError('CPU entrypoint check forbids production network')
 socket.create_connection=no_network;socket.socket.connect=no_network
 for scenario in ['positive','stale-transport','wrong-manifest','wrong-fixture','wrong-profile-pin','wrong-owner-index','wrong-final-nonce','quality-owner-index','quality-cap','quality-clock']:
  root=O/scenario;root.mkdir();source=S;pin=expected
  if scenario in ['stale-transport','wrong-manifest','wrong-fixture','wrong-profile-pin']:
   source=root/'source';shutil.copytree(S,source)
   if scenario=='stale-transport':p=source/'wire_client_r206b.py';p.write_bytes(p.read_bytes()+b'\n')
   elif scenario=='wrong-manifest':p=source/'SOURCE_MANIFEST.json';p.write_bytes(p.read_bytes()+b'\n')
   elif scenario=='wrong-fixture':p=source/'benchmark/gsm8k-fixed32.json';p.write_bytes(p.read_bytes()+b'\n')
   else:
    # Deliberately bad trusted test variant: all source member pins valid, external old profile proof must still reject.
    p=source/'INPUT_BINDING.json';v=json.loads(p.read_bytes());v['profile_manifest_SHA256']='0'*64;dump(p,v);mp=source/'SOURCE_MANIFEST.json';mm=json.loads(mp.read_bytes());mm['files_SHA256']['INPUT_BINDING.json']=sha(p);dump(mp,mm);pin=sha(mp)
  for name in ['source_pin_r281','client_owner_r281']:sys.modules.pop(name,None)
  sys.path[0]=str(source);os.environ['DEEP_SOURCE_MANIFEST_SHA']=pin;step='CPU_SYNTHETIC_'+scenario;report=root/('Deep-latest-full-owner-r285-'+step+'.json');ident=root/'identity.json';identity={'actual_step':step,'Native_SHA256':b['library_SHA256'],'TP':8,'PP':1,'cards':8,'rows':1,'capacity':32768,'source_manifest_SHA256':pin,'prepared_entry_identity':{'upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'}};dump(ident,identity);owner=dict(identity,requests=[{'request_id':'PRIOR_SYNTHETIC_%02d'%i,'admitted_index':i}for i in range(34)]);dump(report,owner);calls=[]
  class Response:
   status=200;headers={'content-type':'application/json'}
   def __init__(self,raw):self.raw=raw
   def read(self):return self.raw
   def __enter__(self):return self
   def __exit__(self,*a):pass
  def fake_url(req,timeout=None):
   body=json.loads(req.data);i=len(calls);assert i<68 and timeout>0;calls.append(copy.deepcopy(body));messages=body['messages'];thinking=body['chat_template_kwargs']['enable_thinking'];budget=body['max_tokens'];assert budget==(8192 if thinking else 4096)and not body['stream'];session=body['user'];ids=[7,8,1];finish='stop';nativefinish='eos_token'
   if i<4:
    assert body['parallel_tool_calls']and body['tools'][0]['function']['name']=='read_file';prompt=9000
    if i%2==0:
     assert body['tool_choice']=='required';finish='tool_calls';message={'role':'assistant','content':'','tool_calls':[{'id':'CPU-opaque-%d-%d'%(i,k),'type':'function','function':{'name':'read_file','arguments':json.dumps({'path':path})}}for k,path in enumerate(['left.json','right.json'])]}
    else:
     assert body['tool_choice']=='none';previous=next(v['tool_calls']for v in messages if v.get('tool_calls'));tools=[v for v in messages if v['role']=='tool'];assert[v['tool_call_id']for v in tools]==list(reversed([v['id']for v in previous]));byid={v['tool_call_id']:json.loads(v['content'])for v in tools};answer={}
     for call in previous:
      data=byid[call['id']];name=json.loads(call['function']['arguments'])['path'];values={v['id']:v['value']for v in data['records']};answer[name]={'nonce':data['nonce'],'sum':sum(values[k]for k in data['query_ids'])}
     answer['total']=sum(v['sum']for v in answer.values())
     if scenario=='wrong-final-nonce':answer['left.json']['nonce']='WRONG'
     message={'role':'assistant','content':json.dumps(answer)}
   else:
    assert not thinking and body['tool_choice']=='none';prompt=900;name='gsm8k-fixed32.json'if i<36 else'mmlu-fixed32.json';cases=json.loads((S/'benchmark'/name).read_bytes())['cases'];offset=i-4 if i<36 else i-36;case=cases[offset]
    if i<36:assert messages[-1]['content']==case['question'];answer=case['gold_numeric']
    else:assert messages[-1]['content']==case['question']+chr(10)+chr(10).join('%s. %s'%(chr(65+k),v)for k,v in enumerate(case['choices']));answer=chr(65+case['answer'])
    message={'role':'assistant','content':'#### '+answer}
    if scenario=='quality-cap'and i==4:finish='length';nativefinish='length';ids=[7]*4096
   if thinking:message['reasoning_content']='CPU_SYNTHETIC_REASONING'
   actualindex=34+i;wrong=1 if(scenario=='wrong-owner-index'and i==0)or(scenario=='quality-owner-index'and i==4)else 0;rid='CPU_SYNTHETIC_%03d'%i;rc=[hashlib.sha256((v.get('reasoning_content')or v.get('reasoning')).encode()).hexdigest()for v in messages if v.get('role')=='assistant'and(v.get('reasoning_content')or v.get('reasoning'))];start=100.+actualindex*10;ticks=[start+.2+k*.001 for k in range(len(ids))];end=ticks[-1]+.1
   if scenario=='quality-clock'and i==4:ticks[0]=float('nan')
   h={'qualified':True,'prequeue_bound':True,'markers_sent_to_Native':False,'prompt_tokens':prompt,'generation_prefix_correct':True,'generation_thinking':thinking,'source_RC_SHA256_sequence':rc,'RC_one_contribution_per_source_field':True,'RC_order_preserved':True,'RC_contributions_nonoverlapping':True,'content_one_contribution_each_and_ordered':True,'all_content_contribution_slices_equal':True,'actual_prompt_SHA256':'CPU_SYNTHETIC','Native_prompt_token_ids_SHA256':'CPU_SYNTHETIC'}
   pre=[{'kind':'prefill','past':k,'tokens':min(1024,prompt-k),'total':min(prompt,k+1024),'emitted':k+1024>=prompt}for k in range(0,prompt,1024)];row={'request_id':rid,'admitted_index':actualindex+wrong,'prompt_tokens':prompt,'max_new_tokens':budget,'normal_reset_before':True,'normal_reset_after':True,'native_finish_reason':nativefinish,'generated_token_ids':ids,'actual_generated_tokens':len(ids),'Native_continuation':pre+[{'kind':'decode_summary','calls':len(ids)-1}],'backend_token_monotonic':ticks,'started_monotonic':start,'ended_monotonic':end,'Native_forward_host_summary':{'prefill_calls':len(pre),'prefill_s':.01*len(pre),'decode_calls':len(ids)-1,'decode_s':.001*(len(ids)-1)},'admission':{'diagnostic_user':session,'history_render_audit':h,'accepted_monotonic':start-.1},'final_metadata':{'CPU_SYNTHETIC':True}};owner['requests'].append(row);dump(report,owner);usage={'prompt_tokens':prompt,'completion_tokens':len(ids),'total_tokens':prompt+len(ids)};return Response(json.dumps({'id':rid,'choices':[{'message':message,'finish_reason':finish}],'usage':usage}).encode())
  urllib.request.urlopen=fake_url
  def run_main(filename,out):
   m=module(source/filename,'CPU_'+filename.replace('.py','')+'_'+scenario);old=sys.argv;sys.argv=[str(source/filename),'--base-url','http://cpu.invalid/v1','--model','DeepSeek-v4','--output-dir',str(out),'--identity',str(ident),'--service-report',str(report),'--expected-step',step,'--manifest-sha',pin,'--deadline-unix',str(time.time()+8000),'--timeout','7200']
   try:rc=m.main()
   finally:sys.argv=old
   return rc,json.loads((out/'result.json').read_bytes())
  rc,mc=run_main('client_multicall_r281.py',root/'multicall');quality=None;qr=None
  if mc['passed']:qr,quality=run_main('client_quality_r281.py',root/'quality')
  case={'name':scenario,'real_multicall_returncode':rc,'multicall_passed':mc['passed'],'multicall_error':mc.get('error'),'quality_returncode':qr,'quality_passed':quality.get('passed')if quality else None,'quality_error':quality.get('error')if quality else None,'synthetic_HTTP':len(calls)};print(json.dumps(case),flush=True)
  if scenario=='positive':assert rc==0 and mc['passed']and qr==0 and quality['passed']and len(calls)==68 and len(owner['requests'])==102;assert[quality['suites'][v]['correct']for v in ['GSM8K','MMLU']]==[32,32]
  elif scenario in ['stale-transport','wrong-manifest','wrong-fixture','wrong-profile-pin']:assert rc==1 and not mc['passed']and len(calls)==0
  elif scenario in ['wrong-owner-index','wrong-final-nonce']:assert rc==1 and not mc['passed']and len(calls)==2
  elif scenario in ['quality-owner-index','quality-clock']:assert mc['passed']and qr==1 and not quality['passed']and len(calls)==5
  else:assert mc['passed']and qr==1 and not quality['passed']and len(calls)==68 and quality['all64_requests_completed']and quality['suites']['GSM8K']['caps']==1 and quality['suites']['MMLU']['count']==32
  case['passed']=True;q['cases'].append(case)
 assert 'torch'not in sys.modules and 'infinilm'not in sys.modules;q['passed']=len(q['cases'])==10
except BaseException as e:q.update(error=repr(e),failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(e.__traceback__)])
finally:
 urllib.request.urlopen=realurl;q['completed']=True;dump(O/'CPU-prepared.json',q);print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error','failure_locations']}),flush=True)
assert q['passed']
