"""Reuse the qualified finite owned-child supervisor, for one VGGT GPU."""
import datetime,hashlib,importlib.metadata as metadata,json,os,shutil,signal,socket,stat,subprocess,sys,time
from pathlib import Path
from deadline_contract import schedule,remaining,TERM_GRACE_SECONDS
from owned_processes import snapshot,refresh,cleanup
W=Path(__file__).resolve().parent.parent;S=W/'src';F=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-native-observer-CPU-v1/framework')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def record(name,v):
 with (W/name).open('x') as f:json.dump(v,f,indent=2)
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=12).strip().splitlines()
def admission(binding):
 rows=[[v.strip() for v in x.split(',')] for x in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'])]
 assert len(rows)==8 and {r[0]:r[1] for r in rows}==binding['UUID_by_index'] and {r[0]:r[2] for r in rows}==binding['PCI_by_index'];r=[r for r in rows if r[0]=='0'][0];assert r[3]=='NVIDIA A100-SXM4-40GB'
 apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']);selected=[x for x in apps if x.split(',')[0].strip()==r[1]]
 return dict(card=r,compute_apps=selected,other_compute_apps_count=len(apps)-len(selected),all_idle=not selected and int(r[5])<=16 and int(r[6])==0,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
def main():
 intent=json.loads((W/'intent.json').read_bytes());epoch=intent['epoch'];clock=schedule(epoch,time.time());mono=time.monotonic();stop_remaining=clock['supervisor_stop']-time.time();assert stop_remaining>700
 record('supervisor-intent.json',dict(claim=intent['claim'],request_id=intent['request_id'],**clock,PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])))
 def expired(*_):raise TimeoutError('original supervisor hard deadline exhausted')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,epoch+890-time.time())
 for name,pin in intent['files'].items():assert sha(S/name)==pin
 prior=json.loads((S/'PRIOR_CPU.json').read_bytes());C=Path(prior['root']);assert sha(C/'CPU-result.json')==prior['CPU_result_SHA256'] and sha(C/'framework-closure.json')==prior['framework_closure_SHA256']
 qualification=json.loads((C/'CPU-result.json').read_bytes());assert qualification['passed'] and qualification['shared_tests']==7 and len(qualification['native_contract_rejected'])==12
 mf=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(mf['files'])==321 and mf['commit']=='5d12acb6fc4693243acca44cd27ede1b24567889'
 for row in mf['files']:
  p=F/row['name'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['SHA256'] and stat.S_IMODE(p.stat().st_mode)==row['mode']
 upstream=Path('/home/whj/nvidia-baseline/sources/vggt');up=json.loads((S/'UPSTREAM_FILES.json').read_bytes())
 for row in up['files']:
  p=upstream/row['name'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['SHA256']
 ref=json.loads((S/'ORIGINAL_REFERENCE.json').read_bytes());assert sha(Path(ref['CPU_result_path']))==ref['CPU_result_SHA256'];assets=json.loads(Path(ref['CPU_result_path']).read_bytes());w=Path(assets['model_weight']['path']);assert w.is_file() and not w.is_symlink() and w.stat().st_size==assets['model_weight']['bytes'] and w.with_name('model.pt.source-sha256').read_text().strip()==ref['model_weights_SHA256']
 assert sha(S/'NATIVE_BINDING.json')==prior['binding_SHA256'] and shutil.disk_usage(W).free>=40*1024**3
 versions={k:metadata.version(k) for k in ['torch','numpy','accelerate','huggingface-hub']};assert versions==qualification['versions']
 assert not any(k in os.environ for k in ['RANK','LOCAL_RANK','WORLD_SIZE','GROUP_RANK'])
 binding=json.loads((S/'GPU_BINDING.json').read_bytes());gate=admission(binding)
 if not gate['all_idle']:record('prelaunch-result.json',dict(state='PRELAUNCH_OCCUPIED',GPU_launched=False,admission=gate));return
 remaining(epoch,clock['deadline'],time.time());assert time.time()<epoch+150
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',VGGT_FIRST_UPDATE_EPOCH=str(epoch),VGGT_FIRST_UPDATE_DEADLINE=str(clock['native_deadline']),VGGT_FIRST_UPDATE_BINDING=str(S/'NATIVE_BINDING.json'),VGGT_FIRST_UPDATE_BINDING_SHA256=prior['binding_SHA256'],VGGT_FIRST_UPDATE_OUTPUT=str(W/'capture'))
 argv=[sys.executable,'-B','-u',str(S/'run_native25.py'),str(W/'train')];record('gpu-intent.json',dict(claim=intent['claim'],request_id=intent['request_id'],**clock,admission=gate,GPU_cards=1,command=argv,versions=versions,framework_commit=mf['commit'],framework_files=321,whole_native25=True,formal_performance=False,raw_export_bytes=0,preflight_seconds=time.monotonic()-mono))
 registry={};failure=None;timed_out=False;child=None
 try:
  with (W/'native.private.log').open('xb') as log:
   child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row=snapshot().get(child.pid)
   if row:registry[child.pid]={k:row[k] for k in ['PID','PPID','PGID','startticks','state']}
   record('child-owner.json',dict(PID=child.pid,identity=registry.get(child.pid)))
   while child.poll() is None:
    refresh(registry,W);p=W/'owned-registry.tmp';p.write_text(json.dumps(registry));p.replace(W/'owned-registry.json')
    if time.time()>=clock['supervisor_stop']:timed_out=True;break
    try:child.wait(timeout=min(2,max(.1,clock['supervisor_stop']-time.time())))
    except subprocess.TimeoutExpired:pass
 except BaseException as error:failure=type(error).__name__
 finally:
  if child is not None and child.poll()==0:
   settle=min(clock['supervisor_stop']+10,time.time()+10)
   while refresh(registry,W) and time.time()<settle:time.sleep(.2)
  cleaned=cleanup(registry,W,TERM_GRACE_SECONDS)
  if child is not None:
   try:child.wait(timeout=1)
   except subprocess.TimeoutExpired:failure=failure or 'OWNED_ROOT_REMAINS'
 try:post=admission(binding)
 except BaseException as error:post=dict(state='POST_UNKNOWN',error_type=type(error).__name__,compute_apps=None)
 result=W/'train/result.json';capture=W/'capture/progress.json';summary=None
 if capture.is_file():
  v=json.loads(capture.read_bytes());summary={k:v[k] for k in ['phase','completed_global_steps','first_update_complete','failed','artifact_bytes','diagnostic_seconds']};summary['files']=len(v['files'])
 actual=dict(claim=intent['claim'],request_id=intent['request_id'],state='NATIVE_RETURNED',**clock,GPU_launched=child is not None,native_returncode=child.returncode if child else None,timed_out=timed_out,controller_error=failure,cleanup=cleaned,post=post,owners=registry,wall_seconds=time.time()-epoch,native_log_SHA256=sha(W/'native.private.log') if (W/'native.private.log').is_file() else None,result_SHA256=sha(result) if result.is_file() else None,capture_progress_SHA256=sha(capture) if capture.is_file() else None,capture=summary)
 actual['keeper_passed']=bool(child and child.returncode==0 and not timed_out and failure is None and not cleaned['termination_needed'] and cleaned['owned_processes_remaining']==0 and post['compute_apps']==[]);record('outcome.json',actual);print(json.dumps(actual))
if __name__=='__main__':main()
