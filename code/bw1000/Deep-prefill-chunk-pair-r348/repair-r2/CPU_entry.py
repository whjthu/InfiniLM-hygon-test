"""Execute actual six-request client and real serve()/cleanup with CPU fakes.
The mature loader is source/AST/dependency checked; no Native dlopen or weights.
"""
import pathlib,json,hashlib,os,sys,time,types,importlib.util,ast,threading,subprocess,copy,urllib.request,traceback
from types import SimpleNamespace as NS
import numpy as np
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;S=D/'owner';O=R/'candidates/Deep-prefill-chunk-pair-r348r2-CPU';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
q=dict(completed=False,passed=False,CPU_only=True,GPU=False,Native_dlopen=False,weights_opened=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],cases=[])
try:
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--owner-sha',required=True);a=ap.parse_args();assert sha(S/'SOURCE_MANIFEST.json')==a.owner_sha;mf=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in mf['files_SHA256'].items());sys.path.insert(0,str(S));os.environ['DEEP_SOURCE_MANIFEST_SHA']=a.owner_sha
 from native_binding_r206b import binding
 b,receipt,norm=binding();assert b['prefill_chunks_supported']==[1024,2048]and b['library_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'and len(norm['link']['full']['objects'])==48
 from chunk_contract import descriptor_for_index,ARMS
 assert [descriptor_for_index(b,i)['prefill_chunk']for i in range(6)]==list(ARMS)
 for bad in [-1,6,True,'1']:
  try:descriptor_for_index(b,bad)
  except ValueError:pass
  else:raise AssertionError('bad descriptor index accepted')
 import preweight_r206b,bootstrap_r206b
 os.environ['DEEP_R277_GPUSTART_UNIX']='0'
 try:preweight_r206b.run()
 except TimeoutError:pass
 else:raise AssertionError('expired preweight reached GPU')
 tree,proof=bootstrap_r206b.transform(bootstrap_r206b.BASE.read_bytes());assert proof['Native_forward_AST_unchanged']and proof['original_finally_destroy_retained']and proof['weight_open_and_load_deadline_hooks']=={'safe_open':1,'modern_native_engine_load':1};q['actual_loader_AST']=proof
 from native_decode_r206b import generate
 reasons=NS(LENGTH='length',EOS_TOKEN='eos_token',CANCELED='canceled')
 for chunk in [1024,2048]:
  for cancel in ['none','before','midprefill','expired']:
   os.environ['DEEP_R277_GPUSTART_UNIX']=str(time.time())if cancel!='expired'else'0';req=NS(prompt_token_ids=[7]*8489,max_new_tokens=128,cancel_event=threading.Event(),abort_reason=None,emitted=[]);req.emit_token=req.emitted.append
   if cancel=='before':req.cancel_event.set()
   calls=[]
   def fw(ids,pos,rows,past,total,offsets):
    calls.append([len(ids),past[0],total[0]]);assert pos==list(range(past[0],total[0]))and offsets==[0,len(ids)]
    if cancel=='midprefill':req.cancel_event.set()
    return 0,None,[1 if total[0]>=8491 else 8 if total[0]==8490 else 7]
   try:reason,trace=generate(req,fw,lambda z:None,lambda z:None,reasons,chunk_size=chunk)
   except TimeoutError:assert cancel=='expired'and not calls
   else:
    if cancel=='none':assert reason=='eos_token'and req.emitted==[7,8,1]and sum(x['tokens']for x in trace if x['kind']=='prefill')==8489
    else:assert reason=='canceled'and not req.emitted
   q['cases'].append(dict(kind='actual_generate',chunk=chunk,cancel=cancel,passed=True,calls=len(calls)))
 # Actual client main + actual pinned WireClient, with only HTTP transport faked.
 import client_chunk_pair
 fixture=json.loads((S/'frozen-toolprompt.json').read_text());expected=fixture['expected_answer'];realurl=urllib.request.urlopen
 for scenario in ['success','wrong-binding','wrong-answer','cap128','short-prompt']:
  root=O/('client-'+scenario);root.mkdir();tag='CPU348-'+scenario;report=root/('Deep-prefill-chunk-pair-r348r2-'+tag+'.json');ident=root/'identity.json';identity=dict(actual_step=tag,Native_SHA256=b['library_SHA256'],TP=8,PP=1,cards=8,rows=1,capacity=32768,source_manifest_SHA256=a.owner_sha,prepared_entry_identity={'upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'});dump(ident,identity);owner=dict(identity,requests=[]);dump(report,owner);calls=[]
  class Response:
   status=200;headers={'content-type':'application/json'}
   def __init__(self,v):self.v=v
   def read(self):return self.v
   def __enter__(self):return self
   def __exit__(self,*args):pass
  def fakeurl(request,timeout):
   body=json.loads(request.data);i=len(calls);assert i<6 and body['max_tokens']==128 and body['messages']==fixture['body']['messages']and body['tools']==fixture['body']['tools'];calls.append(body);chunk=ARMS[i];prompt=8489 if scenario!='short-prompt'else 8000;ids=[7,8,1]if scenario!='cap128'else[7]*128;start=100+i*10;ticks=[start+.5+k*.001 for k in range(len(ids))];history=dict(qualified=True,prequeue_bound=True,markers_sent_to_Native=False,prompt_tokens=prompt,Native_prompt_token_ids_SHA256=fixture['actual_prompt_token_ids_SHA256']);trace=[dict(kind='prefill',past=k,tokens=min(chunk,prompt-k),total=min(prompt,k+chunk),emitted=k+chunk>=prompt)for k in range(0,prompt,chunk)];r=dict(request_id='CPU348-'+str(i),admitted_index=i,max_new_tokens=128,normal_reset_before=True,normal_reset_after=True,prompt_tokens=prompt,admission={'diagnostic_user':body['user'],'history_render_audit':history,'accepted_monotonic':start-.1},native_finish_reason='eos_token'if scenario!='cap128'else'length',generated_token_ids=ids,actual_generated_tokens=len(ids),backend_token_monotonic=ticks,started_monotonic=start,ended_monotonic=ticks[-1]+.1,Native_continuation=trace,Native_forward_host_summary=dict(prefill_calls=len(trace),prefill_s=.1,decode_calls=len(ids)-1,decode_s=.01),prefill_binding=descriptor_for_index(b,i),final_metadata={'CPU_SYNTHETIC':True})
   if scenario=='wrong-binding':r['prefill_binding']['prefill_chunk']=4096
   owner['requests'].append(r);dump(report,owner);answer=copy.deepcopy(expected)
   if scenario=='wrong-answer':answer['total']+=1
   return Response(json.dumps({'id':r['request_id'],'choices':[{'message':{'role':'assistant','content':json.dumps(answer)},'finish_reason':'stop'if scenario!='cap128'else'length'}],'usage':{'prompt_tokens':prompt,'completion_tokens':len(ids),'total_tokens':prompt+len(ids)}}).encode())
  urllib.request.urlopen=fakeurl;oldargv=sys.argv;sys.argv=['client_chunk_pair.py','--base-url','http://cpu.invalid/v1','--model','Deep','--output-dir',str(root/'out'),'--identity',str(ident),'--service-report',str(report),'--manifest-sha',a.owner_sha,'--expected-step',tag,'--deadline-unix',str(time.time()+2000)]
  try:rc=client_chunk_pair.main()
  finally:sys.argv=oldargv;urllib.request.urlopen=realurl
  assert (rc==0)==(scenario=='success');assert len(calls)==(6 if scenario=='success'else 1);q['cases'].append(dict(kind='actual_client_main',scenario=scenario,passed=True,HTTP=len(calls)))
 # Real worker serve()/cleanup; only GPU/HTTP dependencies and client process fake.
 server_module=types.ModuleType('infinilm.server.inference_server');server_module.__file__=str(R/'candidates/latest-family-service-r270/Deep/python/infinilm/server/inference_server.py');server_module.DEFAULT_STREAM_TIMEOUT=1800;server_module.DEFAULT_REQUEST_TIMEOUT=7200;rootmodule=types.ModuleType('infinilm');rootmodule.__path__=[];serverpkg=types.ModuleType('infinilm.server');serverpkg.__path__=[];rootmodule.server=serverpkg;serverpkg.inference_server=server_module;sys.modules.update({'infinilm':rootmodule,'infinilm.server':serverpkg,'infinilm.server.inference_server':server_module})
 servers=[]
 class Server:
  def __init__(self,*args):self.started=False;self.should_exit=False;self.force_exit=False;servers.append(self)
  def run(self):raise AssertionError('fake HTTP target must not execute')
 sys.modules['uvicorn']=NS(Server=Server,Config=lambda *args,**kwargs:NS(**kwargs))
 import worker_r206b as worker
 for scenario in ['success-six','coldload','client-failure','Native-failure']:
  eng=NS(_admission_lock=threading.Lock(),_fatal=None,finished=0,stopped=False);eng.stop=lambda:setattr(eng,'stopped',True);threads=[];requests=[];children=[];phasepath=[None];tag='CPU348-'+q['actual_step'].replace('.','_')+'-'+scenario
  class Req:
   def __init__(self,i):self.request_id='CPU348-owner-'+str(i);self.prompt_token_ids=[7]*8489;self.max_new_tokens=128;self.cancel_event=threading.Event();self.abort_reason=None;self.generated_token_ids=[];self._admission_metadata_ready=True;self.metadata={'diagnostic_user':f'Deep348-{i}-{ARMS[i]}','history_render_audit':{'qualified':True}};self.finish_reason=None
   def emit_token(self,t):self.generated_token_ids.append(t)
   def finish(self,r):self.finish_reason=r;eng.finished+=1
   def fail(self,e):self.finish_reason='error';eng.finished+=1
  requests=[Req(i)for i in range(6)];eng.take=lambda timeout:requests.pop(0)
  class Thread:
   def __init__(self,*args,**kwargs):self.ident=None;self.alive=False;threads.append(self)
   def start(self):self.ident=1;self.alive=True;servers[-1].started=True
   def is_alive(self):return self.alive
   def join(self,timeout):assert eng.stopped;self.alive=False
  class Child:
   returncode=None
   def poll(self):
    if scenario=='client-failure':raise RuntimeError('CPU injected client failure')
    if eng.finished==6:
     dump(phasepath[0]/'result.json',{'completed':True,'passed':True})
     for i in range(6):(phasepath[0]/f'request-{i}.wire').write_text('CPU_SYNTHETIC')
     self.returncode=0;return 0
    return self.returncode
   def terminate(self):self.returncode=-15
   def kill(self):self.returncode=-9
   def wait(self,timeout):return self.returncode
  def popen(args,**kwargs):
   from client_owner_r281 import bind_identity
   actual_identity=pathlib.Path(args[args.index('--identity')+1]);actual_report=args[args.index('--service-report')+1];actual_manifest=args[args.index('--manifest-sha')+1];actual_step=args[args.index('--expected-step')+1];assert actual_manifest==a.owner_sha
   identity=bind_identity(actual_identity,actual_manifest,actual_report);assert identity['prepared_entry_identity']==prepared['identity']and identity['actual_step']==actual_step
   missing=copy.deepcopy(identity);missing.pop('prepared_entry_identity');badpath=O/('missing-identity-'+scenario+'.json');dump(badpath,missing)
   try:bind_identity(badpath,actual_manifest,actual_report)
   except KeyError as exc:assert exc.args==('prepared_entry_identity',)
   else:raise AssertionError('Missing prepared identity silently accepted')
   negargs=args[2:];negargs[negargs.index('--identity')+1]=str(badpath);negout=O/('missing-identity-client-'+scenario);negargs[negargs.index('--output-dir')+1]=str(negout);oldargv=sys.argv;oldurl=urllib.request.urlopen;http=[0]
   def forbidden_http(*args,**kwargs):http[0]+=1;raise AssertionError('Missing identity reached HTTP')
   urllib.request.urlopen=forbidden_http;sys.argv=negargs
   try:rc=client_chunk_pair.main()
   finally:sys.argv=oldargv;urllib.request.urlopen=oldurl
   failed=json.loads((negout/'result.json').read_text());assert rc==1 and not failed['passed']and failed['completed']and failed['requests']==[]and http==[0]and failed['actual_HTTP_attempts']==failed['actual_HTTP_completed']==failed['actual_Native_observed']==failed['actual_Native_completed']==0
   assert 'prepared_entry_identity'in failed['error']
   q['cases'].append(dict(kind='actual_worker_identity_to_actual_client_bind_identity_and_main',scenario=scenario,positive=True,missing_field_negative=True,negative_HTTP=0,actual_identity_manifest_step_from_worker_argv=True,passed=True))
   phasepath[0]=pathlib.Path(args[args.index('--output-dir')+1]);phasepath[0].mkdir();c=Child();children.append(c);return c
  worker.threading=NS(Thread=Thread);worker.subprocess=NS(Popen=popen,STDOUT=subprocess.STDOUT);os.environ['DEEP_R277_GPUSTART_UNIX']='0'if scenario=='coldload'else str(time.time());state={'past':0}
  def reset(ctx):state['past']=0;return 0
  def forward(ids,pos,rows,past,total,offsets):
   if scenario=='Native-failure':raise RuntimeError('CPU injected Native failure')
   assert past[0]==state['past'];state['past']=total[0];return 0,np.zeros((1,129280),dtype=np.uint16),[1 if total[0]>=8491 else 8 if total[0]==8490 else 7]
  prepared={'engine':eng,'FinishReason':reasons,'server':object(),'app':object(),'tokenizer':object(),'identity':{'upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0','eos_token_ids':[1]}};ns={'__latest_prepared_r277__':prepared,'tag':tag,'torch':NS(cuda=NS(mem_get_info=lambda i:(29*1024**3,64*1024**3))),'np':np,'lib':NS(modern_native_engine_reset=reset),'ctx':None,'check':lambda v:None,'forward':forward,'cfg':{'vocab_size':129280},'model':'/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8','__bootstrap_proof_r206b__':{'CPU_mock':True},'__provider_proof_r206b__':{'CPU_mock':True},'__create_proof_r206b__':{'CPU_mock':True}}
  error=None
  try:worker.serve(ns)
  except BaseException as e:error=repr(e)
  actual=json.loads((R/'results'/('Deep-prefill-chunk-pair-r348r2-'+tag+'.json')).read_text());assert eng.stopped and not threads[0].is_alive()and not actual['cleanup']['HTTP_thread_alive_after_cleanup'];assert (error is None)==(scenario=='success-six'),error
  if scenario=='success-six':assert actual['passed']and actual['actual_HTTP']==actual['actual_Native']==6 and actual['normal_server_shutdown']and [r['prefill_binding']['prefill_chunk']for r in actual['requests']]==list(ARMS)
  if scenario=='coldload':assert threads[0].ident is None and not children
  q['cases'].append(dict(kind='actual_worker_serve_cleanup',scenario=scenario,passed=True,error=error,Native=len(actual['requests']),normal_server_shutdown=actual['normal_server_shutdown']))
 q['passed']=True
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['completed']=True;dump(O/'CPU-entry.json',q);print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error']}));assert q['passed']
