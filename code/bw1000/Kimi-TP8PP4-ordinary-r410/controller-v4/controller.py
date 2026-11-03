"""347 single-target lifecycle with404 observation-loss and always-post closure."""
from pathlib import Path
import argparse,datetime,json,math,os,re,socket,subprocess,time,traceback
from contract_r410 import TARGET_NODES,CPU_NODE,GPU_SECONDS,NATIVE_SECONDS,SHELL_SECONDS,CPU_SECONDS,RUNTIME_SHA,NATIVE_SHA,CPU825_SHA,CPU826_SHA,TERMINAL,sha,publish_once,validate_plan,verify_bundle,environment
from slurm_backend import SlurmBackend
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=Path(__file__).resolve().parent
def verify_sources(root,directory,plan,execution=True):
 validate_plan(plan,execution);verify_bundle(directory,plan['controller_source_manifest_SHA256'])
 runtime=Path(root)/'incoming/Kimi-TP8PP4-ordinary-r410/runtime-v3';verify_bundle(runtime,RUNTIME_SHA)
 for path,digest in [('candidates/Kimi-TP8PP4-ordinary-r410-CPU-v4/CPU-result.json',CPU825_SHA),('candidates/Kimi-TP8PP4-ordinary-r410-runtime-CPU-v1/CPU-result.json',CPU826_SHA)]:
  p=Path(root)/path;assert sha(p)==digest;q=json.loads(p.read_bytes());assert q['completed']and q['passed']and q['GPU']==0 and not q['Native_dlopen']
  assert q.get('native_SHA256',q.get('Native_SHA256'))==NATIVE_SHA
 evidence=json.loads((Path(directory)/'STATIC_PROVIDER_EVIDENCE.json').read_bytes())
 assert evidence['runtime_source_manifest_SHA256']==RUNTIME_SHA and evidence['Native_SHA256']==NATIVE_SHA and evidence['dynamic_provider_qualified']is True and evidence['dynamic_provider_actual_GPU']=='910380.566'
 assert sha(runtime/'CAPI_PROVIDERS.json')==evidence['provider_map_SHA256']
 assert sha(Path(directory)/'ELF_provider_evidence.json')==evidence['ELF_evidence_SHA256']
 specs=json.loads((runtime/'CAPI_PROVIDERS.json').read_bytes())['symbols'];assert sorted(specs)==evidence['symbols']and len(specs)==20
 from collections import Counter
 assert sorted(Counter(v['path']for v in specs.values()).values())==[1,19]
 elf=json.loads((Path(directory)/'ELF_provider_evidence.json').read_bytes());assert elf['Native_dlopen']is False and elf['ELF_only']is True
 for path,entry in elf['providers'].items():
  assert sha(Path(root)/path.removeprefix('/refactor/'))==entry['SHA256']
  exported={line.split()[-1]for line in entry['symbols']}
  for name,spec in specs.items():
   if spec['path']==path:assert name in exported and spec['SHA256']==entry['SHA256']
 log=evidence['CPU836'];assert log['state']=='FAILED'and log['receipt_exists']is False and sha(Path(root)/log['log'])==log['log_SHA256']
 proof=evidence['prior_dynamic_proof'];assert proof['actual_GPU']=='910380.566'and proof['Model_passed']is False
 assert sha(runtime/'native_binding.py')==proof['binder_SHA256']and sha(runtime/'CAPI_PROVIDERS.json')==proof['provider_map_SHA256']
 assert len(proof['reports'])==4 and {v['node']for v in proof['reports']}==set(range(4))
 for entry in proof['reports']:
  p=Path(entry['path']);assert sha(p)==entry['SHA256'];q=json.loads(p.read_bytes());assert q['actual_step']=='910380.566'and q['node']==entry['node']and q['Native_SHA256']==NATIVE_SHA
  assert set(q['actual_CAPI_providers'])==set(specs)
  for name,spec in specs.items():
   assert q['actual_CAPI_providers'][name]==spec['path']
   actual=q['Native_SHA256']if spec['path'].endswith('/libKimi_TP8PP4_ordinary_r410.so')else q['actual_loaded_libraries']['libinfinilm_runtime.so']['SHA256'];assert actual==spec['SHA256']
 assert sha(runtime/'local-affinity-CPU-result.json')==evidence['affinity_fix']['local_CPU_receipt_SHA256']

 lib=Path(root)/'candidates/Kimi-TP8PP4-ordinary-r410-CPU-v4/libKimi_TP8PP4_ordinary_r410.so';assert sha(lib)==NATIVE_SHA
 return runtime
def state(v):return v['row'][2].split()[0].rstrip('+')if v.get('row')else None
def terminal_success(v):return state(v)=='COMPLETED'and v['row'][3]=='0:0'and v['srun_exit_code']==0
def terminal_failure(v):return v['srun_exit_code']not in[None,0]or state(v)in TERMINAL and(state(v)!='COMPLETED'or v['row'][3]!='0:0')
class Controller:
 def __init__(self,root,plan,backend,nonce,clock=time.time,sleep=time.sleep):
  self.root=Path(root);self.plan=plan;self.backend=backend;self.nonce=nonce;assert len(nonce)>=32;self.clock=clock;self.sleep=sleep
  prefix='Kimi-ordinary-r410-'+plan['run'];self.watch=self.root/'results'/(prefix+'-watch.json');self.cancel=self.root/'control'/(prefix+'-cancel.json');self.claim=self.root/'control'/(prefix+'-claim.json')
  self.q=dict(completed=False,passed=False,GPU_submitted=False,run=plan['run'],publication_commit=plan['publication_commit'],runtime_source_manifest_SHA256=RUNTIME_SHA,controller_source_manifest_SHA256=plan['controller_source_manifest_SHA256'],target_cards=32,draft_cards=0,retry=False,reload=False,Model_loaded=None,checkpoint_IO=None)
  self.epoch=None;self.exit_seen=None;self.terminal_seen=None
  self.keeper_start=float(os.environ.get('KIMI410_KEEPER_START_UNIX',self.clock()));assert math.isfinite(self.keeper_start)
 def save(self):
  self.q['probe_records']=getattr(self.backend,'probe_records',[]);self.q['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=self.watch.with_suffix('.tmp');p.write_text(json.dumps(self.q,indent=2,allow_nan=False)+'\n');p.replace(self.watch)
 def observation(self):
  v=self.backend.observe();self.q['last_observation']=v
  if v['row']:self.q['actual_GPU']=v['row'][0]
  if v['srun_exit_code']is not None and state(v)not in TERMINAL:
   if self.exit_seen is None:self.exit_seen=self.clock()
   if self.clock()-self.exit_seen>90:raise RuntimeError('Owned srun exit but accounting unresolved; no replay')
  if state(v)in TERMINAL and v['srun_exit_code']is None:
   if self.terminal_seen is None:self.terminal_seen=self.clock()
   if self.clock()-self.terminal_seen>90:raise RuntimeError('Accounting terminal but srun exit unresolved; no replay')
  return v
 def run(self):
  validate_plan(self.plan,True);assert not any(p.exists()for p in[self.watch,self.cancel,self.claim]);publish_once(self.watch,self.q)
  try:
   self.q['actual_final_CLI_preflight']=self.backend.final_preflight();self.save()
   self.backend.validate_resources();before=self.backend.idle('before');self.q['fresh_before']=before;self.backend.no_foreign()
   heads={v['node']:v['ib_ipv4'][0]for v in before};assert set(heads)==set(TARGET_NODES)
   self.epoch=self.clock();assert math.isfinite(self.epoch)
   assert self.keeper_start+CPU_SECONDS-self.epoch>=GPU_SECONDS+300,'Insufficient original CPU budget; no Model load or sliding deadline'
   self.q.update(original_GPU_start_unix=self.epoch,GPU_deadline_unix=self.epoch+GPU_SECONDS,Native_deadline_unix=self.epoch+NATIVE_SECONDS,shell_deadline_unix=self.epoch+SHELL_SECONDS,original_keeper_start_unix=self.keeper_start,keeper_deadline_unix=self.keeper_start+CPU_SECONDS)
   publish_once(self.claim,dict(Model_GO=True,GPU_GO=True,runtime_source_manifest_SHA256=RUNTIME_SHA,Native_SHA256=NATIVE_SHA,target_cards=32,draft_cards=0,Native_seconds=NATIVE_SECONDS,publication_commit=self.plan['publication_commit'],original_GPU_start_unix=self.epoch))
   self.q['submission']=self.backend.start(environment(self.plan,self.nonce,self.epoch,heads[TARGET_NODES[0]],self.cancel,self.claim),self.epoch);self.q['GPU_submitted']=True;self.save()
   while self.clock()<self.epoch+GPU_SECONDS:
    if self.cancel.exists():raise RuntimeError('Original cancellation')
    try:v=self.observation()
    except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as e:
     self.q.setdefault('transient_observation_errors',[]).append(dict(at_unix=self.clock(),error=repr(e),actual_GPU=self.backend.actual))
     if self.backend.proc.poll()not in[None,0]:raise RuntimeError('Owned srun nonzero during observation loss')
     self.save();self.sleep(min(15,max(0,self.epoch+GPU_SECONDS-self.clock())));continue
    if terminal_failure(v):raise RuntimeError('Original target failed; no retry/reload')
    if terminal_success(v):break
    self.save();self.sleep(min(15,max(0,self.epoch+GPU_SECONDS-self.clock())))
   else:raise TimeoutError('Original5400s bound; never restart')
   # Always run independent passive post even if terminal artifact visibility fails.
   audit_error=None
   try:
    from collect import collect
    self.q['final_evidence']=collect(self.root,self.plan['run'],self.backend.actual,time.monotonic()+90);self.q.update(Model_loaded=True,checkpoint_IO=True)
   except BaseException as e:audit_error=e;self.q['terminal_artifact_error']=repr(e)
   self.q['post']=self.backend.idle('post');self.q['post_empty']=True
   if audit_error is not None:raise audit_error
   self.q.update(completed=True,passed=True,state='TWO_ORDINARY_REQUESTS_NORMAL_CLEAN_PENDING_ROOT_QUALITY_AND_COST_AUDIT')
  except BaseException as e:
   self.q.update(error=repr(e),error_frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)for f in traceback.extract_tb(e.__traceback__)[-8:]],completed=True,passed=False,state='FAILED_OR_UNKNOWN_NO_RETRY')
   if self.q['GPU_submitted']:
    if not self.cancel.exists():publish_once(self.cancel,dict(run=self.plan['run'],cancelled=True,reason=type(e).__name__))
    self.q['owned_forced_cleanup']=self.backend.cancel_owned()
    if not self.q.get('post_empty'):
     try:self.q['post_after_failure']=self.backend.idle('failure-post');self.q['post_empty']=True
     except BaseException as close:self.q['post_error']=repr(close)
  finally:self.save()
  return self.q
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--preflight-output',type=Path);a=p.parse_args();plan=json.loads(a.plan.read_bytes())
 verify_sources(R,D,plan,execution=not a.preflight)
 if a.preflight:
  from final_preflight import run
  q=run(Path('/refactor'),D,plan);q['controller_source_manifest_SHA256']=plan['controller_source_manifest_SHA256']
  if a.preflight_output:publish_once(a.preflight_output,q)
  print(json.dumps(q));return 0
 assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]==CPU_NODE
 assert 'KIMI410_KEEPER_START_UNIX'in os.environ
 q=Controller(R,plan,SlurmBackend(R,D,plan),os.environ['KIMI410_CONTROLLER_NONCE']).run();print(json.dumps({k:q.get(k)for k in['completed','passed','actual_GPU','state']}));return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
