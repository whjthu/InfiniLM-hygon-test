"""Validate binding and loader AST; no actual-root/extension import or GPU job."""
import pathlib,sys,os,json,hashlib,ast,importlib.util,datetime,traceback,time,threading,signal
from types import SimpleNamespace
signal.alarm(205)
R=pathlib.Path('/refactor');D=R/'incoming/Deep-owner-static-r280';S=R/'incoming/Deep-latest-full-owner-r277';O=R/'candidates/Deep-owner-static-r280';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Torch_imported':False,'actual_root_import':False,'Native_dlopen':False,'Model':0,'weights_opened':False,'no_new_compile_or_link':True,'GPU_GO':False}
try:
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert sha(D/'SOURCE_MANIFEST.json')==a.manifest_sha;mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in mf['files_SHA256'].items());assert sha(S/'SOURCE_MANIFEST.json')==mf['owner_source_manifest_SHA256'];sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items());q['owner_source_manifest_SHA256']=mf['owner_source_manifest_SHA256']
 for n in sm['files_SHA256']:
  if n.endswith('.py'):ast.parse((S/n).read_bytes(),filename=str(S/n))
 sys.path.insert(0,str(S));import native_binding_r206b as binding
 b,receipt,normalized=binding.binding();m=binding.manifest();assert b['rows']==1 and b['capacity']==32768 and b['prefill_chunk']==1024;assert normalized['derived_binding']and normalized['original_guard_proof_preserved']and len(normalized['link']['full']['objects'])==48
 assert m['DSO_SHA256']==b['library_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 q['binding']={'Native_SHA256':b['library_SHA256'],'CPU600_SHA256':b['CPU_receipt_SHA256'],'original_guard_CPU_SHA256':b['original_guard_binding']['CPU_receipt_SHA256'],'objects':len(normalized['link']['full']['objects']),'actual_DTK_bindings':normalized['actual_DTK_bindings']}
 import bootstrap_r206b as bootstrap
 raw=bootstrap.BASE.read_bytes();assert hashlib.sha256(raw).hexdigest()==bootstrap.BASE_SHA;tree,proof=bootstrap.transform(raw);q['loader_AST']=proof;assert proof['Native_forward_AST_unchanged']and proof['original_finally_destroy_retained'];compile(tree,str(bootstrap.BASE),'exec')
 # Same-process preweight source ordering; this CPU step does not import it.
 pre=ast.parse((S/'preweight_r206b.py').read_bytes());fn=next(x for x in pre.body if isinstance(x,ast.FunctionDef)and x.name=='run');positions={}
 for x in ast.walk(fn):
  if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=='set_device':positions['assigned_GPU0']=x.lineno
  if isinstance(x,ast.Import)and any(v.name=='infinilm'for v in x.names):positions['actual_root']=x.lineno
  if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='prepare_app':positions['actual_processor_server']=x.lineno
  if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=='CDLL'and x.args and not isinstance(x.args[0],ast.Constant):positions['Native_dlopen']=x.lineno
 assert positions['assigned_GPU0']<positions['actual_root']<positions['actual_processor_server']<positions['Native_dlopen'];q['future_preweight_order']=positions
 worker=(S/'worker_r206b.py').read_text();assert "prepared=ns['__latest_prepared_r277__']"in worker and 'Phase failed; stop without advancing or retry'in worker and 'DEEP_R277_GPUSTART_UNIX'in worker
 # CAPI interface remains the original forward; synthetic continuation only.
 import native_decode_r206b as decode
 os.environ['DEEP_R277_GPUSTART_UNIX']=str(time.time());emitted=[];calls=[];samples=iter([7,1]);req=SimpleNamespace(prompt_token_ids=[10,11],max_new_tokens=3,cancel_event=threading.Event(),emit_token=emitted.append);reasons=SimpleNamespace(EOS_TOKEN='eos',LENGTH='length',CANCELED='cancel')
 def forward(*args):calls.append(args);return 0,None,[next(samples)]
 reason,trace=decode.generate(req,forward,lambda rc:None,lambda logits:None,reasons);assert reason=='eos'and emitted==[7,1]and len(calls)==2 and trace[0]['tokens']==2;before=len(calls);os.environ['DEEP_R277_GPUSTART_UNIX']=str(time.time()-30000)
 try:decode.generate(req,forward,lambda rc:None,lambda logits:None,reasons)
 except TimeoutError:q['expired_absolute_deadline_rejected_before_forward']=len(calls)==before
 else:raise AssertionError('Expired original residency deadline accepted')
 assert q['expired_absolute_deadline_rejected_before_forward'];assert 'torch'not in sys.modules and 'infinilm'not in sys.modules;q['synthetic_continuation']={'prefill_tokens':2,'real_model':False,'CAPI_called':False,'calls':2,'EOS_handling':True};q['passed']=True
except BaseException as e:q.update(error=repr(e),failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(e.__traceback__)])
finally:
 q['completed']=True;(O/'CPU-prepared.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error','failure_locations']}),flush=True)
assert q['passed']
