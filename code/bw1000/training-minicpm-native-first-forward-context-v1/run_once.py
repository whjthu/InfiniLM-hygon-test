from pathlib import Path
import hashlib,importlib.metadata as metadata,json,os,shutil,signal,socket,subprocess,time
from owned_processes import snapshot,refresh,cleanup
ROOT=Path(__file__).resolve().parent.parent;SRC=ROOT/'src'
intent=json.loads((ROOT/'intent.json').read_bytes());epoch=intent['epoch'];deadline=intent['deadline'];assert deadline==epoch+1200
runtime=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1')
def left(reserve=0):
 value=deadline-time.time()-reserve;assert value>0;return value
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):left(40);h.update(raw)
 return h.hexdigest()
def save(name,v):
 with (ROOT/name).open('x') as f:json.dump(v,f,indent=2)
def admission(binding):
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'],text=True,timeout=min(12,left(35))).strip().splitlines();rows=[[x.strip() for x in line.split(',')] for line in cards]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=min(12,left(35))).strip().splitlines()
 assert len(rows)==8 and {int(x[0]):x[1] for x in rows}=={int(k):v for k,v in binding['UUID_by_index'].items()}
 return dict(cards=rows,compute_apps=apps,all_idle=not apps and all(int(x[5])<=16 and int(x[6])==0 for x in rows),observed_epoch=time.time())
def main():
 def expired(*_):raise TimeoutError('CPU preflight exhausted')
 signal.signal(signal.SIGALRM,expired);signal.alarm(min(120,max(1,int(left(200)))))
 installed=json.loads((ROOT/'installed.json').read_bytes())
 for n,pin in installed['files'].items():assert sha(SRC/n)==pin,n
 for row in json.loads((SRC/'SOURCE_CONTEXT.json').read_bytes())['files']:
  source=Path(row['path']);assert source.stat().st_size==row['bytes'] and sha(source)==row['SHA256']
 config=json.loads((SRC/'CONFIG.json').read_bytes());binding=json.loads((SRC/'GPU_BINDING.json').read_bytes());first=admission(binding)
 if not first['all_idle']:save('prelaunch-result.json',dict(state='PRELAUNCH_OCCUPIED',GPU_launched=False,admission=first));return
 original=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-masks-reference-v1');launch=json.loads((SRC/'original-launch.json').read_bytes());assert sha(SRC/'original-launch.json')=='015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789'
 for n,pin in [('MODEL_FILES.json','ddcb8756dc66474f348bd81bb1730874ba78af8b46567e6a99596cedf5594ae6'),('GPU_BINDING.json','d28da2f48d8f26b9f7b217f366fa296c060c089cc18bd243239d42f3574aa8a3')]:assert sha(SRC/n)==pin
 model=json.loads((SRC/'MODEL_FILES.json').read_bytes())
 for n,pin in model['files'].items():
  p=Path(model['model_root'])/n;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],n
 proof=json.loads((SRC/'SOURCE_PROOF.json').read_bytes());expected={**proof['pinned_public_Swift_sources'],'transformers/trainer.py':proof['pinned_public_Trainer_source_SHA256']}
 for role,pin in expected.items():assert sha(Path(metadata.distribution('transformers' if role.startswith('transformers/') else 'ms-swift').locate_file(role)))==pin,role
 rb=json.loads((SRC/'RUNTIME_BINDING.json').read_bytes());assert sha(runtime/'direct-v2/receipt.json')==rb['metadata_overlay_receipt_SHA256'] and sha(runtime/'triton-v1/receipt.json')==rb['triton_receipt_SHA256']
 for role,pin in json.loads((original/'capture/rank-00/runtime-providers.json').read_bytes()).items():assert sha(Path(pin['path']))==pin['SHA256'],role
 source=Path(metadata.distribution('transformers').locate_file('transformers/models/qwen3_5/modeling_qwen3_5.py'));assert sha(source)==config['model_source_SHA256']
 data=Path(launch['command'][launch['command'].index('--dataset')+1]);assert sha(data)=='0e476394e3b0117d90cfad8f89ccfafe1800917692e86efed84eaf182c453811'
 assert sha(Path('/home/whj/nvidia-baseline/swift_callback_nvidia_baseline_v2.py'))=='d731f9889c22381514c2181c27f99ac86051930ab2e7792958db5b34e34f27e8'
 for row in config['ranks']:
  for kind in ('initial','input','output','mask'):
   p=Path(row[kind]['path']);assert p.stat().st_size==row[kind]['bytes'] and sha(p)==row[kind]['SHA256']
 assert shutil.disk_usage(ROOT).free>=16*1024**3
 versions={n:metadata.version(n) for n in ('torch','torchvision','transformers','ms-swift','peft','accelerate','tokenizers')};assert versions['transformers']=='5.12.1' and versions['ms-swift']=='4.4.2' and versions['peft']=='0.19.1'
 import torch
 assert str(torch.__version__)=='2.8.0a0+5228986c39.nv25.06' and not torch.cuda.is_initialized()
 argv=launch['command'].copy();argv[0]='/home/whj/nvidia-baseline/env-vlm/bin/swift';argv[argv.index('--output_dir')+1]=str(ROOT/'train')
 i=argv.index('--external_plugins')+1
 while i<len(argv) and not argv[i].startswith('--'):i+=1
 argv.insert(i,str(SRC/'framework/vlm/plugins/native_first_forward_context.py'))
 i=argv.index('--callbacks')+1;j=i
 while j<len(argv) and not argv[j].startswith('--'):j+=1
 assert argv[i:j]==['nvidia_baseline'];argv[i:j]=['native_first_forward_context','nvidia_baseline']
 environment=os.environ.copy();environment.update(launch['environment']);environment['PATH']=environment['PATH'].replace('/home/whj/nvidia-baseline/env-vlm-fast/bin:','/home/whj/nvidia-baseline/env-vlm/bin:',1);environment['PATH']=str(runtime/'bin')+os.pathsep+environment['PATH'];environment['PYTHONPATH']=str(runtime/'triton-v1/site')+os.pathsep+str(runtime/'site')+os.pathsep+str(SRC/'framework')
 port=29881;environment.update(OUTPUT_DIR=str(ROOT/'train'),MASTER_PORT=str(port),NATIVE_FORWARD_CONFIG=str(SRC/'CONFIG.json'),NATIVE_FORWARD_CONFIG_SHA256=sha(SRC/'CONFIG.json'),NATIVE_FORWARD_EPOCH=str(epoch),NATIVE_FORWARD_DEADLINE=str(epoch+1080),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 assert environment['NPROC_PER_NODE']=='8' and environment['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
 with socket.socket() as probe:probe.bind(('127.0.0.1',port))
 final=admission(binding)
 if not final['all_idle']:save('prelaunch-result.json',dict(state='PRELAUNCH_OCCUPIED_AFTER_CPU',GPU_launched=False,admission=final));return
 signal.alarm(0);left(200)
 native_phase_start_ns=time.monotonic_ns()
 save('gpu-intent.json',dict(native_phase_start_monotonic_ns=native_phase_start_ns,claim=intent['claim'],epoch=epoch,deadline=deadline,native_deadline=epoch+1080,supervisor_stop=epoch+1110,versions=versions,command=argv,admission=final,GPU_cards=8,optimizer_updates=0,planned_native_forwards_per_rank=1,framework=json.loads((SRC/'FRAMEWORK.json').read_bytes()),formal_performance=False))
 registry={};child=None;timed=False;error=None
 try:
  with (ROOT/'native.log').open('xb') as log:
   child=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python',*argv],env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   row=snapshot().get(child.pid)
   if row:registry[child.pid]={k:row[k] for k in ('PID','PPID','PGID','startticks','state')}
   save('child-owner.json',dict(PID=child.pid,identity=registry.get(child.pid)))
   while child.poll() is None:
    refresh(registry,ROOT);p=ROOT/'owned-registry.tmp';p.write_bytes(json.dumps(registry).encode());p.replace(ROOT/'owned-registry.json')
    if time.time()>=epoch+1110:timed=True;break
    try:child.wait(timeout=min(2,max(.1,epoch+1110-time.time())))
    except subprocess.TimeoutExpired:pass
 except BaseException as e:error=dict(type=type(e).__name__,message=str(e)[:300])
 finally:
  if child is not None and child.poll()==0:
   until=min(epoch+1130,time.time()+8)
   while refresh(registry,ROOT) and time.time()<until:time.sleep(.2)
  cleaned=cleanup(registry,ROOT,10)
  if child is not None:
   try:child.wait(timeout=1)
   except subprocess.TimeoutExpired:error=error or dict(type='OWNED_ROOT_REMAINS')
 post=admission(binding);results=[];failures=[]
 for rank in range(8):
  root=ROOT/'frames'/('rank-%02d'%rank)
  if (root/'result.json').exists():
   raw=(root/'result.json').read_bytes();results.append(dict(rank=rank,SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw)))
  if (root/'failure.json').exists():
   raw=(root/'failure.json').read_bytes();failures.append(dict(rank=rank,SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw)))
 actual=dict(native_phase_wall_seconds=(time.monotonic_ns()-native_phase_start_ns)/1e9,native_phase_timing='Native launch through cleanup/post/metadata collection, including load, dataset preparation and diagnostic audit; not exclusive GPU or formal throughput',claim=intent['claim'],epoch=epoch,deadline=deadline,GPU_launched=child is not None,native_returncode=child.returncode if child else None,timed_out=timed,error=error,cleanup=cleaned,post=post,owners=registry,native_log_SHA256=sha(ROOT/'native.log'),rank_results=results,rank_failures=failures,optimizer_updates=0,raw_tensor_transfer_bytes=0,Model_GO=False)
 actual['keeper_passed']=bool(child and child.returncode==0 and not timed and error is None and not cleaned['termination_needed'] and cleaned['owned_processes_remaining']==0 and post['compute_apps']==[] and len(results)==8 and not failures and all(r['value']['passed'] for r in results))
 save('outcome.json',actual)
if __name__=='__main__':main()
