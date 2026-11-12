from pathlib import Path
import base64,datetime,hashlib,json,subprocess,uuid
B=Path(__file__).parent;D=B/'training-hymotion-original-metadata-v1';A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');original=json.loads((D/'original-launch.json').read_bytes());request=uuid.uuid4().hex
REMOTE=r'''
from pathlib import Path
import base64,hashlib,json,re,subprocess
import yaml
cfg=ORIGINAL['config'];S=Path(ORIGINAL['source']);V=Path(ORIGINAL['weights'])/cfg['variant'];files={}
def small(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_size>200000:files[str(p)]=dict(read=False,exists=p.is_file(),is_symlink=p.is_symlink());return None
 raw=p.read_bytes();files[str(p)]=dict(read=True,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode());return raw
model=yaml.safe_load(small(V/'config.yml'));M=Path(cfg['training_data_manifest']);raw=M.read_bytes();assert len(raw)<200000 and hashlib.sha256(raw).hexdigest()==cfg['training_data_manifest_sha256'];manifest=json.loads(raw)
shards=[]
for row in manifest['shards']:
 p=M.parent/row['path'];item=dict(name=row['path'],expected_SHA256=row['sha256'],sample_count=row['sample_count'],exists=p.is_file(),is_symlink=p.is_symlink(),within_root=p.resolve().is_relative_to(M.parent.resolve()))
 if p.is_file():
  stat=p.stat();item.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,device=stat.st_dev)
 shards.append(item)
dataset=M.parent/manifest['dataset_manifest'];stat=dataset.stat();assert dataset.is_file() and stat.st_size<1000000;dataset_sha=hashlib.sha256(dataset.read_bytes()).hexdigest()
small(Path('/mnt/data/nvidia-baseline-20261010/cuda_source/generation/src/gencuda/backends/hy_motion_training.py'));small(S/'hymotion/utils/loaders.py')
classes={key:str(model[key]) for key in ['train_pipeline','network_module']};need={str(value).rsplit('.',1)[-1] for value in classes.values()};found={};pins={};head=None;dirty=None;git_available=True;omitted=[];total=0
try:
 head=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True,timeout=10).strip();dirty=subprocess.check_output(['git','-C',str(S),'diff','--name-only'],text=True,timeout=10).splitlines();names=subprocess.check_output(['git','-C',str(S),'ls-files','-z','--','*.py'],timeout=10).split(b'\0')
 for n in names:
  if not n:continue
  name=n.decode();p=S/name
  if len(pins)>=500 or not p.is_file() or p.is_symlink() or p.stat().st_size>1000000 or total+p.stat().st_size>8*1024**2:omitted.append(name);continue
  raw=p.read_bytes();total+=len(raw);pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
  for identity in need:
   if identity.isidentifier() and re.search(rb'^class\s+'+identity.encode()+rb'\b',raw,re.M):found.setdefault(identity,[]).append(name);small(p)
except subprocess.SubprocessError:git_available=False
p=V/'latest.ckpt';checkpoint=dict(path=str(p),exists=p.is_file(),is_symlink=p.is_symlink(),body_read=False)
if p.is_file():
 stat=p.stat();checkpoint.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,device=stat.st_dev)
print(json.dumps(dict(request_id=REQUEST,requested_frames=128,original_frames=manifest['frames'],original_feature_dim=manifest['feature_dim'],original_sample_count=manifest['sample_count'],requested_frames_match=manifest['frames']==128,CPU_GO=False,GPU_GO=False,Model_GO=False,files=files,original_config=dict(classes=classes,train_pipeline_args=model['train_pipeline_args'],network_module_args=model['network_module_args']),checkpoint=checkpoint,training_manifest=dict(path=str(M),bytes=len(raw) if False else M.stat().st_size,SHA256=hashlib.sha256(M.read_bytes()).hexdigest(),shards=shards,dataset_manifest_SHA256=dataset_sha,dataset_manifest_SHA_matches=dataset_sha==manifest['dataset_manifest_sha256']),source_git=dict(available=git_available,head=head,dirty_paths=dirty,Python_files=pins,Python_bytes=total,omitted_paths=omitted,class_definitions=found),checkpoint_body_bytes_read=0,cache_tensor_body_bytes_read=0,raw_sample_or_condition_transfer_bytes=0,framework_imported=False,GPU_initialized=False)))
'''
payload=('ORIGINAL='+repr(original)+'\nREQUEST='+repr(request)+'\n').encode()+REMOTE.encode();(D/'final-source-identity-request.json').open('x').write(json.dumps(dict(request_id=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),metadata_only=True,frame_conflict_preserved=True,weight_or_cache_tensor_read=False)))
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','/home/whj/nvidia-baseline/env/bin/python -B -u -'];r=subprocess.run(ssh,input=payload,capture_output=True,timeout=45)
for n,raw in [('final-source-identity.stdout.private.json',r.stdout),('final-source-identity.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
v=json.loads(r.stdout) if r.returncode==0 else None
if v:
 for index,(n,pin) in enumerate(v['files'].items()):
  if pin.get('read'):
   raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];p=D/'identity-public-source'/str(index)/Path(n).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
record=dict(returncode=r.returncode,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),value=v,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
with (D/'original-final-source-and-asset-identity.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(dict(returncode=r.returncode,original_frames=v.get('original_frames') if v else None,original_sample_count=v.get('original_sample_count') if v else None,checkpoint_bytes=v.get('checkpoint',{}).get('bytes') if v else None,cache_files=len(v['training_manifest']['shards']) if v else None,cache_bytes=sum(x.get('bytes',0) for x in v['training_manifest']['shards']) if v else None,source_Python_files=len(v['source_git']['Python_files']) if v else None,source_classes=v['original_config']['classes'] if v else None,CPU_GO=False,GPU_GO=False)))
