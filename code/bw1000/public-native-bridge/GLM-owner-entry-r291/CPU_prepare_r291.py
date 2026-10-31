"""Actual GLM client mains and transport. Synthetic responses only at urllib boundary."""
import pathlib,json,hashlib,sys,os,signal,shutil,datetime,time,importlib.util,urllib.request,socket,copy,ast,traceback
from types import SimpleNamespace
signal.alarm(205)
R=pathlib.Path('/refactor');D=R/'incoming/GLM-owner-entry-r291';S=R/'incoming/GLM-latest-full-owner-r290';O=R/'candidates/GLM-owner-entry-r291';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10))
def module(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'production_HTTP':0,'synthetic_owner_and_quality_only':True,'real_multicall_main_quality_main_transport_and_validators':True,'unchanged_normal35_not_rerun':True,'actual_GLМ_or_Native_execution':False,'cases':[]}
realurl=urllib.request.urlopen
try:
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert sha(D/'SOURCE_MANIFEST.json')==a.manifest_sha;mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in mf['files_SHA256'].items());expected=mf['owner_source_manifest_SHA256'];assert sha(S/'SOURCE_MANIFEST.json')==expected;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items());q['owner_source_manifest_SHA256']=expected
 for n in sm['files_SHA256']:
  if n.endswith('.py'):ast.parse((S/n).read_bytes(),filename=str(S/n))
 # The four formerly checked loader/guard components remain immutable apart from intended edits.
 sys.path.insert(0,str(S));os.environ['GLM_SOURCE_MANIFEST_SHA']=expected
 # Actual preweight dependency import and expired-deadline boundary; never import root on CPU.
 import importlib.abc
 class NoGPUImport(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0]in ['torch','infinilm','_infinilm']:raise RuntimeError('CPU preweight dependency import forbids '+fullname)
 guard=NoGPUImport();sys.meta_path.insert(0,guard);os.environ['GLM_R290_GPUSTART_UNIX']='0'
 import preweight_r290 as preweight
 try:preweight.validate()
 except TimeoutError:pass
 else:raise AssertionError('Expired preweight reached Torch/root initialization')
 assert 'torch'not in sys.modules and 'infinilm'not in sys.modules and '_infinilm'not in sys.modules;q['actual_preweight_guarded_import_and_original_deadline_PASS']=True
 from native_chain_r186 import verify as native_verify
 b=native_verify();q['original_Native_chain_reused']=b
 base=R/'incoming/modern_glm_full_model_probe_r2.py';assert sha(base)=='a9b643a21e5ff5eaea021d768aa0342a155948fd2f962157c08a683647d06a05';text=base.read_text();loader='def load(ctx, report):'+chr(10)+text[text.index(" sdk=sorted("):text.index(' def control(x):')]
 from loader_guard_r290 import guard_loader
 original=ast.parse(loader);tree,proof=guard_loader(loader);hooks=[v for v in ast.walk(tree)if isinstance(v,ast.Subscript)and isinstance(v.value,ast.Tuple)and v.value.elts and isinstance(v.value.elts[0],ast.Call)and isinstance(v.value.elts[0].func,ast.Name)and v.value.elts[0].func.id=='__weight_deadline_check_r290__'];assert len(hooks)==3
 from deadline_r290 import check as check_deadline
 for hook in hooks:
  try:eval(compile(ast.fix_missing_locations(ast.Expression(hook)),'<actual-GLM-weight-deadline-hook>','eval'),{'__weight_deadline_check_r290__':check_deadline})
  except TimeoutError:pass
  else:raise AssertionError('Weight hook evaluated original operation before expiry rejection')
 class Undo(ast.NodeTransformer):
  def visit_Subscript(self,node):
   if isinstance(node.value,ast.Tuple)and len(node.value.elts)==2 and isinstance(node.value.elts[0],ast.Call)and isinstance(node.value.elts[0].func,ast.Name)and node.value.elts[0].func.id=='__weight_deadline_check_r290__':return self.visit(node.value.elts[1])
   return self.generic_visit(node)
 assert ast.dump(Undo().visit(copy.deepcopy(tree)),include_attributes=False)==ast.dump(original,include_attributes=False)
 q['loader_AST']={'weight_calls':proof,'expired_before_original_weight_operations':True,'original_loader_AST_retained_after_unwrap':True,'original_BASE_SHA256':sha(base)}
 plan=json.loads((S/'PREPARATION.json').read_bytes());assert plan['planned_HTTP']==121 and plan['planned_Native']==120 and(plan['resource_plan']['ordinary_budget'],plan['resource_plan']['thinking_budget'])==(4096,8192)
 # Real cleanup helper: initialization may fail before server/thread allocation.
 import cleanup_owned_r290 as cleanup
 import io
 from types import SimpleNamespace
 class Engine:
  def __init__(self,events):self.events=events;self.closed=False
  def stop(self):self.closed=True;self.events.append('engine.stop')
 events=[];engine=Engine(events);log=io.StringIO();c=cleanup.cleanup_owned(None,None,None,engine,log);assert engine.closed and log.closed and not c['errors']and not c['normal_server_shutdown']and not c['forced_cleanup'];q['cleanup_before_thread_allocation']=c
 # A synthetic known session's parent exits on TERM but its child survives until group KILL.
 events=[];engine=Engine(events);log=io.StringIO();state={'parent':True,'child':True}
 class Client:
  pid=1000123
  def poll(self):return None if state['parent']else -15
  def wait(self,timeout):events.append('parent.wait');assert not state['parent'];return -15
 def fake_killpg(pgid,sig):
  assert pgid==1000123
  events.append('group.TERM'if sig==signal.SIGTERM else'group.KILL');state['parent']=False
  if sig==signal.SIGKILL:state['child']=False
 original_killpg=cleanup.os.killpg;cleanup.os.killpg=fake_killpg
 try:c=cleanup.cleanup_owned(Client(),SimpleNamespace(),None,engine,log,owned_pgid=1000123)
 finally:cleanup.os.killpg=original_killpg
 assert events==['engine.stop','group.TERM','parent.wait','group.KILL','parent.wait']and not state['child']and log.closed and c['forced_cleanup']and not c['normal_server_shutdown']and not c['errors'];q['cleanup_parent_exits_TERM_child_requires_group_KILL']={'passed':True,'events':events,'synthetic_process_group_only':True,'real_processes_signalled':0}
 def no_network(*a,**k):raise RuntimeError('CPU entrypoint check forbids production network')
 socket.create_connection=no_network;socket.socket.connect=no_network
 for scenario in ['positive','stale-transport','wrong-manifest','wrong-fixture','wrong-profile-pin','wrong-owner-index','wrong-final-nonce','quality-owner-index','quality-cap','quality-clock','wrong-family-EOS','wrong-effective-prefix','pair-wrong-window','pair-wrong-prompt']:
  root=O/scenario;root.mkdir();source=S;pin=expected
  if scenario in ['stale-transport','wrong-manifest','wrong-fixture','wrong-profile-pin']:
   source=root/'source';shutil.copytree(S,source)
   if scenario=='stale-transport':p=source/'wire_client_r290.py';p.write_bytes(p.read_bytes()+b'\n')
   elif scenario=='wrong-manifest':p=source/'SOURCE_MANIFEST.json';p.write_bytes(p.read_bytes()+b'\n')
   elif scenario=='wrong-fixture':p=source/'benchmark/gsm8k-fixed32.json';p.write_bytes(p.read_bytes()+b'\n')
   else:
    # Deliberately bad trusted test variant: all source member pins valid, external old profile proof must still reject.
    p=source/'INPUT_BINDING.json';v=json.loads(p.read_bytes());v['profile_manifest_SHA256']='0'*64;dump(p,v);mp=source/'SOURCE_MANIFEST.json';mm=json.loads(mp.read_bytes());mm['files_SHA256']['INPUT_BINDING.json']=sha(p);dump(mp,mm);pin=sha(mp)
  for name in ['source_pin_r290','client_owner_r290']:sys.modules.pop(name,None)
  sys.path[0]=str(source);os.environ['GLM_SOURCE_MANIFEST_SHA']=pin;step='CPU_SYNTHETIC_'+scenario;report=root/('GLM-latest-full-owner-r290-'+step+'-stage3-rank0.json');ident=root/'identity.json';identity={'actual_step':step,'run':step,'Native_SHA256':b['library_SHA256'],'TP':8,'PP':4,'cards':32,'rows':1,'capacity':32768,'source_manifest_SHA256':pin,'prepared_entry_identity':{'upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'}};dump(ident,identity);owner=dict(identity,service_requests=[{'request_id':'PRIOR_SYNTHETIC_%02d'%i,'admitted_index':i}for i in range(34)]);dump(report,owner);calls=[]
  class Response:
   status=200;headers={'content-type':'application/json'}
   def __init__(self,raw):self.raw=raw
   def read(self):return self.raw
   def __enter__(self):return self
   def __exit__(self,*a):pass
  def fake_url(req,timeout=None):
   body=json.loads(req.data);i=len(calls);assert i<86 and timeout>0;calls.append(copy.deepcopy(body));messages=body['messages'];thinking=body['chat_template_kwargs'].get('enable_thinking',False);budget=body['max_tokens'];assert budget==(8192 if thinking else 4096)and not body['stream'];session=body['user'];ids=[7,8,154820];finish='stop';nativefinish='eos_token'
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
   elif i<68:
    assert not thinking and body['tool_choice']=='none';prompt=900;name='gsm8k-fixed32.json'if i<36 else'mmlu-fixed32.json';cases=json.loads((S/'benchmark'/name).read_bytes())['cases'];offset=i-4 if i<36 else i-36;case=cases[offset]
    if i<36:assert messages[-1]['content']==case['question'];answer=case['gold_numeric']
    else:assert messages[-1]['content']==case['question']+chr(10)+chr(10).join('%s. %s'%(chr(65+k),v)for k,v in enumerate(case['choices']));answer=chr(65+case['answer'])
    message={'role':'assistant','content':'#### '+answer}
    if scenario=='quality-cap'and i==4:finish='length';nativefinish='length';ids=[7]*4096
   else:
    import client_paired_r290 as paired
    raw=(R/'results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json').read_bytes();assert req.data==raw
    args=paired.validate_request(body);prompt=8900;finish='tool_calls';message={'role':'assistant','content':'','tool_calls':[{'id':'CPU-PAIR-OPAQUE-'+str(i),'type':'function','function':{'name':'calculate','arguments':json.dumps(args)}}]}
   if thinking:message['reasoning_content']='CPU_SYNTHETIC_REASONING'
   if scenario=='wrong-family-EOS'and i==0:ids[-1]=1
   actualindex=34+i;wrong=1 if(scenario=='wrong-owner-index'and i==0)or(scenario=='quality-owner-index'and i==4)else 0;rid='CPU_SYNTHETIC_%03d'%i;rc=[hashlib.sha256((v.get('reasoning_content')or v.get('reasoning')).encode()).hexdigest()for v in messages if v.get('role')=='assistant'and(v.get('reasoning_content')or v.get('reasoning'))];start=100.+actualindex*10;ticks=[start+.2+k*.001 for k in range(len(ids))];end=ticks[-1]+.1
   if scenario=='quality-clock'and i==4:ticks[0]=float('nan')
   h={'qualified':True,'prequeue_bound':True,'markers_sent_to_Native':False,'prompt_tokens':prompt,'generation_prefix_correct':True,'generation_thinking':not(scenario=='wrong-effective-prefix'and i==0),'pending_thinking_prefix_open':not(scenario=='wrong-effective-prefix'and i==0),'source_RC_SHA256_sequence':rc,'RC_one_contribution_per_source_field':True,'RC_order_preserved':True,'RC_contributions_nonoverlapping':True,'content_one_contribution_each_and_ordered':True,'all_content_contribution_slices_equal':True,'actual_prompt_SHA256':'CPU_SYNTHETIC','Native_prompt_token_ids_SHA256':'CPU_SYNTHETIC'}
   pre=[{'kind':'prefill','past':k,'tokens':min(1024,prompt-k),'total':min(prompt,k+1024),'emitted':k+1024>=prompt}for k in range(0,prompt,1024)];row={'request_id':rid,'admitted_index':actualindex+wrong,'prompt_tokens':prompt,'max_new_tokens':budget,'normal_reset_before':True,'normal_reset_after':True,'native_finish_reason':nativefinish,'generated_token_ids':ids,'actual_generated_tokens':len(ids),'Native_continuation':pre+[{'kind':'decode_summary','calls':len(ids)-1}],'backend_token_monotonic':ticks,'started_monotonic':start,'ended_monotonic':end,'Native_forward_host_summary':{'prefill_calls':len(pre),'prefill_s':.01*len(pre),'decode_calls':len(ids)-1,'decode_s':.001*(len(ids)-1)},'admission':{'diagnostic_user':session,'history_render_audit':h,'accepted_monotonic':start-.1},'final_metadata':{'CPU_SYNTHETIC':True}};row['prefill_control_window']=([1,2]+[1,2,2,1]*4)[i-68]if i>=68 else 2;row['pair_index']=i-68 if i>=68 else None;row['paired_warmup']=68<=i<70;row['actual_prompt_token_ids_SHA256']='CPU_SYNTHETIC_PROMPT_CHANGED'if scenario=='pair-wrong-prompt'and i==69 else'CPU_SYNTHETIC_PROMPT';
   if scenario=='pair-wrong-window'and i==68:row['prefill_control_window']=2
   owner['service_requests'].append(row);dump(report,owner);usage={'prompt_tokens':prompt,'completion_tokens':len(ids),'total_tokens':prompt+len(ids)};return Response(json.dumps({'id':rid,'choices':[{'message':message,'finish_reason':finish}],'usage':usage}).encode())
  urllib.request.urlopen=fake_url
  def run_main(filename,out):
   m=module(source/filename,'CPU_'+filename.replace('.py','')+'_'+scenario);old=sys.argv;sys.argv=[str(source/filename),'--base-url','http://cpu.invalid/v1','--model','GLM-5.3','--output-dir',str(out),'--identity',str(ident),'--service-report',str(report),'--expected-step',step,'--manifest-sha',pin,'--deadline-unix',str(time.time()+8000),'--timeout','7200']
   try:rc=m.main()
   finally:sys.argv=old
   return rc,json.loads((out/'result.json').read_bytes())
  rc,mc=run_main('client_multicall_r290.py',root/'multicall');quality=paired_result=None;qr=pr=None
  if mc['passed']:qr,quality=run_main('client_quality_r290.py',root/'quality')
  if quality and quality['passed']:pr,paired_result=run_main('client_paired_r290.py',root/'paired')
  case={'name':scenario,'real_multicall_returncode':rc,'multicall_passed':mc['passed'],'multicall_error':mc.get('error'),'quality_returncode':qr,'quality_passed':quality.get('passed')if quality else None,'quality_error':quality.get('error')if quality else None,'synthetic_HTTP':len(calls),'paired_returncode':pr,'paired_passed':paired_result.get('passed')if paired_result else None,'paired_error':paired_result.get('error')if paired_result else None};print(json.dumps(case),flush=True)
  if scenario=='positive':assert rc==0 and mc['passed']and qr==0 and quality['passed']and len(calls)==86 and len(owner['service_requests'])==120 and pr==0 and paired_result['passed']and quality['actual_HTTP_requests']==64 and mc['actual_HTTP_requests']==4 and sum([30,3,2,mc['actual_HTTP_requests'],quality['actual_HTTP_requests'],paired_result['actual_HTTP_requests']])==121;assert[quality['suites'][v]['correct']for v in ['GSM8K','MMLU']]==[32,32]
  elif scenario in ['stale-transport','wrong-manifest','wrong-fixture','wrong-profile-pin']:assert rc==1 and not mc['passed']and len(calls)==0
  elif scenario in ['wrong-owner-index','wrong-final-nonce','wrong-family-EOS','wrong-effective-prefix']:assert rc==1 and not mc['passed']and len(calls)==2
  elif scenario in ['quality-owner-index','quality-clock']:assert mc['passed']and qr==1 and not quality['passed']and len(calls)==5
  elif scenario.startswith('pair-'):assert mc['passed']and quality['passed']and pr==1 and not paired_result['passed']and len(calls)==(69 if scenario=='pair-wrong-window'else 70)
  else:assert mc['passed']and qr==1 and not quality['passed']and len(calls)==68 and quality['all64_requests_completed']and quality['suites']['GSM8K']['caps']==1 and quality['suites']['MMLU']['count']==32
  case['passed']=True;q['cases'].append(case)
 assert 'torch'not in sys.modules and 'infinilm'not in sys.modules;q['passed']=len(q['cases'])==14
except BaseException as e:q.update(error=repr(e),failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(e.__traceback__)])
finally:
 urllib.request.urlopen=realurl;q['completed']=True;dump(O/'CPU-prepared.json',q);print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error','failure_locations']}),flush=True)
assert q['passed']
