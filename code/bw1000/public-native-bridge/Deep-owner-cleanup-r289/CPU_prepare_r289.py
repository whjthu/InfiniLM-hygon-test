"""Only new owner cleanup/coldload branches; reuse CPU610 client matrix by exact bytes."""
import pathlib,json,hashlib,sys,os,datetime,ast,time,types,subprocess,importlib.util,importlib.abc,traceback,signal
from types import SimpleNamespace
signal.alarm(205)
R=pathlib.Path('/refactor');D=R/'incoming/Deep-owner-cleanup-r289';S=R/'incoming/Deep-latest-full-owner-r287';B=R/'incoming/Deep-latest-full-owner-r285';C=R/'incoming/Deep-latest-controller-r283';O=R/'candidates/Deep-owner-cleanup-r289';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+chr(10))
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'production_HTTP':0,'actual_root_import':False,'Native_dlopen':False,'weights_opened':False,'cleanup_cases':[],'client_matrix_repeated':False}
try:
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert sha(D/'SOURCE_MANIFEST.json')==a.manifest_sha;mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in mf['files_SHA256'].items());ownerpin=mf['owner_source_manifest_SHA256'];assert sha(S/'SOURCE_MANIFEST.json')==ownerpin;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items());q['owner_source_manifest_SHA256']=ownerpin
 cp=R/'candidates/Deep-client-entry-r286/CPU-prepared.json';assert sha(cp)=='a43625aa9a68c9146e5c26ae555d9462916455ad08dadfcd9ddf1f50b750b5e5';prior=json.loads(cp.read_bytes());assert prior['passed']and prior['completed']and prior['actual_preweight_and_legacy_dependency_import_guarded_PASS']and len(prior['cases'])==10;assert sha(B/'SOURCE_MANIFEST.json')==prior['owner_source_manifest_SHA256'];bm=json.loads((B/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(B/n)==h for n,h in bm['files_SHA256'].items())
 for n in ['client_multicall_r281.py','client_quality_r281.py','source_pin_r281.py','wire_client_r206b.py','prepared_family_entry_r268.py','prepared_chat.py','prepared_history.py','native_engine_bridge_r268.py','profiles_r206b.py','chat_prepare_r173.py','client_regular_agent.py','client_long_output.py','client_history_r206b.py']:
  assert(S/n).read_bytes()==(B/n).read_bytes(),n
 assert(S/'client_owner_r281.py').read_text().replace('Deep-latest-full-owner-r287-','Deep-latest-full-owner-r285-')==(B/'client_owner_r281.py').read_text();q['inherited_CPU610_client_contract']=True;q['prior_CPU610_receipt_SHA256']=sha(cp)
 class Guard(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0]in ['torch','infinilm','_infinilm']:raise RuntimeError('CPU cleanup test forbids actualGPU/root import '+fullname)
 sys.meta_path.insert(0,Guard());sys.path.insert(0,str(S));os.environ['DEEP_SOURCE_MANIFEST_SHA']=ownerpin;os.environ['DEEP_R277_GPUSTART_UNIX']='0'
 import preweight_r206b as preweight
 try:preweight.run()
 except TimeoutError:pass
 else:raise AssertionError('Expired preweight reached GPU initialization')
 assert 'torch'not in sys.modules and 'infinilm'not in sys.modules;q['guarded_preweight_import_PASS']=True
 import bootstrap_r206b as bootstrap
 assert sha(bootstrap.BASE)==bootstrap.BASE_SHA
 tree,proof=bootstrap.transform(bootstrap.BASE.read_bytes());assert proof['Native_forward_AST_unchanged']and proof['original_finally_destroy_retained']and proof['weight_open_and_load_deadline_hooks']=={'safe_open':1,'modern_native_engine_load':1};q['loader_AST']=proof;hooks=[v for v in ast.walk(tree)if isinstance(v,ast.Subscript)and isinstance(v.value,ast.Tuple)and v.value.elts and isinstance(v.value.elts[0],ast.Call)and isinstance(v.value.elts[0].func,ast.Name)and v.value.elts[0].func.id=='__weights_deadline_check_r287__'];assert len(hooks)==2
 for v in hooks:
  try:eval(compile(ast.fix_missing_locations(ast.Expression(v)),'<actual_loader_deadline_hook>','eval'),{'__weights_deadline_check_r287__':bootstrap.weights_deadline_check})
  except TimeoutError:pass
  else:raise AssertionError('Expired original weight load hook did not reject first')
 q['expired_deadline_rejected_before_both_weight_open_and_load_expressions']=True
 # Fake only GPU/HTTP dependencies. Real serve() and real cleanup helper execute.
 fake_root=types.ModuleType('infinilm');fake_root.__path__=[];fake_serverpkg=types.ModuleType('infinilm.server');fake_serverpkg.__path__=[];fake_server=types.ModuleType('infinilm.server.inference_server');fake_server.__file__=str(R/'candidates/latest-family-service-r270/Deep/python/infinilm/server/inference_server.py');fake_server.DEFAULT_STREAM_TIMEOUT=1800.;fake_server.DEFAULT_REQUEST_TIMEOUT=7200.;fake_root.server=fake_serverpkg;fake_serverpkg.inference_server=fake_server
 sys.modules.update({'infinilm':fake_root,'infinilm.server':fake_serverpkg,'infinilm.server.inference_server':fake_server});fake_uvicorn=types.ModuleType('uvicorn');fake_uvicorn.Config=lambda*args,**kwargs:SimpleNamespace(**kwargs);servers=[]
 class FakeServer:
  def __init__(self,config):self.started=False;self.should_exit=False;self.force_exit=False;servers.append(self)
  def run(self):raise RuntimeError('CPU fake HTTP target must not execute')
 fake_uvicorn.Server=FakeServer;sys.modules['uvicorn']=fake_uvicorn
 import worker_r206b as worker
 for scenario in ['coldload-deadline-before-thread','client-TERM-timeout']:
  events=[];engine=SimpleNamespace(stopped=False)
  def stop():engine.stopped=True;events.append('engine.stop')
  engine.stop=stop;threads=[];children=[];handles=[]
  class FakeThread:
   def __init__(self,*a,**k):self.ident=None;self.alive=False;threads.append(self)
   def start(self):self.ident=1;self.alive=True;servers[-1].started=True;events.append('thread.start')
   def is_alive(self):return self.alive
   def join(self,timeout):assert engine.stopped;events.append('thread.join');self.alive=False
  worker.threading=SimpleNamespace(Thread=FakeThread)
  class FakeChild:
   returncode=None
   def __init__(self):self.polled=False;self.killed=False;children.append(self)
   def poll(self):
    if not self.polled:self.polled=True;raise RuntimeError('CPU_INJECTED_BODY_FAILURE')
    return -9 if self.killed else None
   def terminate(self):events.append('client.TERM')
   def kill(self):events.append('client.KILL');self.killed=True
   def wait(self,timeout):
    events.append('client.wait')
    if not self.killed:raise subprocess.TimeoutExpired('CPU_FAKE_CHILD',timeout)
    return -9
  def popen(args,**kwargs):handles.append(kwargs['stdout']);return FakeChild()
  worker.subprocess=SimpleNamespace(Popen=popen,STDOUT=subprocess.STDOUT);os.environ['DEEP_R277_GPUSTART_UNIX']='0'if scenario.startswith('coldload')else str(time.time());tag='CPU_CLEANUP_'+q['actual_step'].replace('.','_')+'_'+scenario
  prepared={'engine':engine,'FinishReason':SimpleNamespace(LENGTH='length'),'server':SimpleNamespace(),'app':object(),'tokenizer':object(),'identity':{'upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0','eos_token_ids':[1]}};ns={'__latest_prepared_r277__':prepared,'tag':tag,'torch':SimpleNamespace(),'np':SimpleNamespace(),'lib':SimpleNamespace(),'ctx':None,'check':lambda x:None,'model':'/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8','__bootstrap_proof_r206b__':{'CPU_mock':True},'__provider_proof_r206b__':{'CPU_mock':True},'__create_proof_r206b__':{'CPU_mock':True}}
  try:worker.serve(ns)
  except BaseException as e:error=repr(e);q['last_cleanup_owner_exception']=error
  else:raise AssertionError('Injected owner failure was swallowed')
  report=R/'results'/('Deep-latest-full-owner-r287-'+tag+'.json');assert report.is_file(),('Owner failed before protected startup',error);actual=json.loads(report.read_bytes());cleanup=actual['cleanup'];assert engine.stopped and not threads[0].is_alive()and not cleanup['HTTP_thread_alive_after_cleanup']and not actual['passed']and not actual['normal_server_shutdown']and actual['requests']==[];assert all(h.closed for h in handles)
  if scenario.startswith('coldload'):assert threads[0].ident is None and not children and 'Cold load consumed'in error and not cleanup['forced_cleanup']
  else:assert cleanup['client_KILL_sent']and cleanup['forced_cleanup']and error=="RuntimeError('CPU_INJECTED_BODY_FAILURE')"and events.index('engine.stop')<events.index('thread.join')
  q['cleanup_cases'].append({'name':scenario,'error':error,'passed':True,'HTTP_thread_started':threads[0].ident is not None,'engine_stopped':True,'forced_cleanup':cleanup['forced_cleanup'],'normal_shutdown_not_falsely_claimed':True,'events':events,'actual_serve_report_SHA256':sha(report),'synthetic_dependencies':True,'Native_calls':0})
 q['actual_worker_serve_and_cleanup_executed']=True;q['passed']=len(q['cleanup_cases'])==2
except BaseException as e:q.update(error=repr(e),failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(e.__traceback__)])
finally:q['completed']=True;dump(O/'CPU-prepared.json',q);print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error','failure_locations']}),flush=True)
assert q['passed']
