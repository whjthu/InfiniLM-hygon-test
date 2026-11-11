"""One owned native eight-A100 Gemma reference,using the shared training framework."""
import datetime,hashlib,importlib.metadata as metadata,json,os,shutil,signal,socket,stat,subprocess,sys,time
from pathlib import Path
from deadline_contract import schedule,TERM_GRACE_SECONDS
from owned_processes import snapshot,refresh,cleanup
ROOT=Path(__file__).resolve().parent.parent;SRC=ROOT/'src';FRAMEWORK=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-update-contract-CPU-v1/framework');sys.path.insert(0,str(FRAMEWORK))
from tools.hygon_vlm_recipe import build_launch

def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1024**2),b''):h.update(data)
 return h.hexdigest()
def record(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=12).strip().splitlines()
def admission(binding):
 rows=[[v.strip() for v in x.split(',')] for x in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'])];apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']);assert len(rows)==8 and {r[0]:r[1] for r in rows}==binding['UUID_by_index'] and {r[0]:r[2] for r in rows}==binding['PCI_by_index'];assert all(r[3]=='NVIDIA A100-SXM4-40GB' for r in rows)
 return dict(cards=rows,compute_apps=apps,all_idle=not apps and all(int(r[5])<=16 and int(r[6])==0 for r in rows),observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
def bindable():
 try:
  with socket.socket() as s:s.bind(('127.0.0.1',29891))
  return True
 except OSError:return False

def main():
 epoch=time.time();clock=schedule(epoch,epoch);mono=time.monotonic();installed=json.loads((ROOT/'installed.json').read_bytes());claim=installed['claim'];request=installed['request_id'];assert not (ROOT/'gpu-intent.json').exists(),'Reconcile the original GPU scope,never replay';record('supervisor-intent.json',dict(claim=claim,request_id=request,**clock,PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])))
 def expired(*_):raise TimeoutError('original supervisor hard deadline exhausted')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,890)
 for name,pin in installed['files'].items():assert file_sha(SRC/name)==pin
 fm=json.loads((SRC/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==314 and fm['framework_commit']=='d544b54972e418a0576d575be21c2df780eeb6df'
 for name,pin in fm['files'].items():
  p=FRAMEWORK/name;assert p.stat().st_size==pin['bytes'] and file_sha(p)==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode']
 original=(SRC/'original-launch.json').read_bytes();assert hashlib.sha256(original).hexdigest()=='6ff6fecf17d423e672c2c660a9f774d67e03a8a79ca17f388853d34ca42a15fc'
 binding=json.loads((SRC/'GPU_BINDING.json').read_bytes());first=admission(binding)
 if not first['all_idle'] or not bindable():record('prelaunch-result.json',dict(state='PRELAUNCH_OCCUPIED',GPU_launched=False,claim=claim,request_id=request,admission=first,master_port_bindable=bindable()));return
 hashes_started=time.monotonic();model=json.loads((SRC/'MODEL_FILES.json').read_bytes());model_root=Path(model['model_root']);verified_model=[]
 for name,pin in model['files'].items():
  p=model_root/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and file_sha(p)==pin['SHA256'];verified_model.append(name)
 assets=json.loads((SRC/'INPUT_FILES.json').read_bytes())
 for pin in assets['files']:
  p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and file_sha(p)==pin['SHA256']
 metadata_sources=json.loads((SRC/'INSTALLED_SOURCE_FILES.json').read_bytes())
 for name,pin in metadata_sources.items():
  package='transformers' if name.startswith('transformers/') else 'ms-swift';p=Path(metadata.distribution(package).locate_file(name));assert p.stat().st_size==pin['bytes'] and file_sha(p)==pin['SHA256']
 versions={k:metadata.version(k) for k in ['torch','torchvision','transformers','ms-swift','peft','accelerate','Pillow','tokenizers']};assert versions['torch']=='2.8.0a0+5228986c39.nv25.6' and versions['transformers']=='5.12.1' and versions['ms-swift']=='4.4.2' and versions['peft']=='0.19.1' and versions['accelerate']=='1.14.0'
 import torch
 assert torch.version.cuda=='12.9' and not torch.version.hip and not torch.cuda.is_initialized()
 assert file_sha(SRC/'GEMMA_NATIVE_BINDING.json')=='7bd0b8e46b164ef179ac59a2525322dfb50b6802e3a277c45113a4e9bc6cc8a1'
 assert file_sha(Path('/home/whj/nvidia-baseline/swift_callback_nvidia_baseline_v2.py'))=='d731f9889c22381514c2181c27f99ac86051930ab2e7792958db5b34e34f27e8'
 assert shutil.disk_usage(ROOT).free>=80*1024**3 and not any(k in os.environ for k in ['RANK','LOCAL_RANK','WORLD_SIZE','GROUP_RANK'])
 launch=build_launch('T02.sft',original,Path('/home/whj/nvidia-baseline/env-vlm/bin/swift'),Path(model['model_root']).parent,Path('/mnt/data/nvidia-baseline-20261010/inputs/vlmdata/vlm_report_coco64.cuda.jsonl'),Path('/home/whj/nvidia-baseline'),ROOT/'train',29891,diagnostic_plugin=SRC/'capture_first_updates.py',diagnostic_callback='native_first_updates',diagnostic_callback_position='before')
 argv=launch['command'];overrides=launch['environment_overrides'];assert overrides['NPROC_PER_NODE']=='8' and overrides['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7';final=admission(binding);assert final['all_idle'] and bindable();assert time.monotonic()-mono<150,'Prelaunch exceeded native scope reserve'
 environment=os.environ.copy();environment.update(overrides);environment['PATH']=json.loads(original)['environment']['PATH'];environment.update(PYTHONPATH=str(FRAMEWORK)+os.pathsep+str(SRC),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',FIRST_UPDATE_EPOCH=str(epoch),FIRST_UPDATE_DEADLINE=str(clock['native_deadline']),FIRST_UPDATE_MAX_SECONDS='900',FIRST_UPDATE_CAPTURE=str(ROOT/'capture'),FIRST_UPDATE_PLATFORM='cuda',FIRST_UPDATE_BINDING=str(SRC/'GEMMA_NATIVE_BINDING.json'),FIRST_UPDATE_BINDING_SHA256=file_sha(SRC/'GEMMA_NATIVE_BINDING.json'))
 record('gpu-intent.json',dict(claim=claim,request_id=request,**clock,admission=final,versions=versions,command=argv,GPU_cards=8,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_files=314,original25_preserved=True,formal_performance=False,model_files_verified=verified_model,model_verification_and_preflight_seconds=time.monotonic()-hashes_started,original_input_files_verified=len(assets['files']),structural_binding_SHA256=file_sha(SRC/'GEMMA_NATIVE_BINDING.json')))
 registry={};timed_out=False;failure=None;child=None
 try:
  with (ROOT/'native.private.log').open('xb') as log:
   child=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python',*argv],env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row=snapshot().get(child.pid)
   if row:registry[child.pid]={k:row[k] for k in ['PID','PPID','PGID','startticks','state']}
   record('child-owner.json',dict(PID=child.pid,identity=registry.get(child.pid),claim=claim,request_id=request))
   while child.poll() is None:
    refresh(registry,ROOT);temp=ROOT/'owned-registry.tmp';temp.write_text(json.dumps(registry));temp.replace(ROOT/'owned-registry.json')
    if time.monotonic()-mono>=850:timed_out=True;break
    try:child.wait(timeout=min(2,max(.1,850-(time.monotonic()-mono))))
    except subprocess.TimeoutExpired:pass
 except BaseException as error:failure=type(error).__name__
 finally:
  if child is not None and child.poll()==0:
   settle=min(mono+870,time.monotonic()+10)
   while refresh(registry,ROOT) and time.monotonic()<settle:time.sleep(.2)
  cleaned=cleanup(registry,ROOT,TERM_GRACE_SECONDS)
  if child is not None:
   try:child.wait(timeout=1)
   except subprocess.TimeoutExpired:failure=failure or 'OWNED_ROOT_REMAINS'
 try:post=admission(binding)
 except BaseException as error:post=dict(state='POST_UNKNOWN',error_type=type(error).__name__,compute_apps=None)
 actual=dict(claim=claim,request_id=request,GPU_launched=child is not None,state='NATIVE_RETURNED',**clock,native_returncode=child.returncode if child else None,timed_out=timed_out,controller_error=failure,wall_seconds=time.monotonic()-mono,cleanup=cleaned,post=post,master_port_bindable=bindable(),owners=registry,native_log_SHA256=file_sha(ROOT/'native.private.log') if (ROOT/'native.private.log').is_file() else None,native25_rank_records=sum((ROOT/'train/nvidia-baseline'/('rank-%02d.json'%i)).is_file() for i in range(8)),capture_rank_records=sum((ROOT/'capture'/('rank-%02d'%i)/'receipt.json').is_file() for i in range(8)))
 actual['keeper_passed']=bool(child and child.returncode==0 and not timed_out and failure is None and not cleaned['termination_needed'] and cleaned['owned_processes_remaining']==0 and post['compute_apps']==[] and actual['master_port_bindable']);record('outcome.json',actual);print(json.dumps(actual))
if __name__=='__main__':main()
