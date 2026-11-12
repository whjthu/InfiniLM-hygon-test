"""Fresh exact admission, with an explicit unique local observation tag."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,time
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,40}',tag)
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
base=claim+':code/bw1000/training-llava-A100-first-updates-v1/'
values={n:json.loads(subprocess.check_output(git+['show',base+n])) for n in ('MODEL_FILES.json','GPU_BINDING.json','INPUT_FILES.json','PRIOR_QUALIFICATIONS.json','INSTALLED_SOURCE_FILES.json')}
remote=('BOUND='+repr(values)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,importlib.metadata as md,json,os,shutil,socket,subprocess,sys
phase='namespace'
try:
 W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-llava-A100-first-updates-v1');assert not W.exists()
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 phase='prior-original-qualifications'
 for q in BOUND['PRIOR_QUALIFICATIONS.json']['qualifications']:
  for pin in q['files']:
   p=Path(q['root'])/pin['name'];assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='model-files';model=BOUND['MODEL_FILES.json'];root=Path(model['model_root'])
 for name,pin in model['files'].items():
  p=root/name;s=p.stat();assert p.is_file() and not p.is_symlink() and s.st_size==pin['bytes']
  if name.endswith('.safetensors'):assert dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)==model['weight_stats'][name]
  else:assert sha(p)==pin['SHA256']
 phase='original-shared-loader'
 pin=BOUND['PRIOR_QUALIFICATIONS.json']['original_loader'];p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='original-inputs'
 for pin in BOUND['INPUT_FILES.json']['files']:
  p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='installed-native-source'
 for name,pin in BOUND['INSTALLED_SOURCE_FILES.json'].items():
  dist={'transformers':'transformers','swift':'ms-swift','peft':'peft'}[name.split('/')[0]];p=Path(md.distribution(dist).locate_file(name));assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='GPU-identity-and-idleness'
 def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=12).strip().splitlines()
 cards=[[v.strip() for v in row.split(',')] for row in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'])]
 apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']);binding=BOUND['GPU_BINDING.json']
 assert len(cards)==8 and {r[0]:r[1] for r in cards}==binding['UUID_by_index'] and {r[0]:r[2] for r in cards}==binding['PCI_by_index']
 idle=not apps and all(r[3]=='NVIDIA A100-SXM4-40GB' and int(r[5])<=16 and int(r[6])==0 for r in cards)
 if not idle:
  result=dict(passed=False,phase=phase,state='REFERENCE_HOST_OCCUPIED',observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_cards=8,compute_app_records=len(apps),compute_apps=apps,cards=cards,namespace_absent=True,GPU_launched=False)
 else:
  phase='port-disk-and-environment'
  with socket.socket() as s:s.bind(('127.0.0.1',29903))
  free=shutil.disk_usage(W.parent).free;assert free>=80*1024**3
  assert not any(k in os.environ for k in ('RANK','LOCAL_RANK','WORLD_SIZE','GROUP_RANK'))
  versions={n:md.version(n) for n in ('torch','torchvision','transformers','ms-swift','peft','accelerate','Pillow','tokenizers')}
  assert versions==binding['versions']
  result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_absent=True,model_files=18,weight_full_SHA_reused_with_exact_stats=4,
   original_input_files=67,installed_native_source_files=len(BOUND['INSTALLED_SOURCE_FILES.json']),GPU_cards=8,cards=cards,compute_apps=apps,master_port_bindable=True,free_disk_bytes=free,versions=versions,GPU_launched=False)
 assert 'torch' not in sys.modules
 result['Torch_imported']=False
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,Torch_imported=False,GPU_launched=False)
print(json.dumps(result))
'''
started=time.monotonic();A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
intent=dict(claim=claim,payload_SHA256=hashlib.sha256(remote).hexdigest(),tag=tag,read_only=True,started_monotonic=started)
with (D/('preflight-'+tag+'-intent.json')).open('x') as f:json.dump(intent,f,indent=2)
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=90);ended=time.monotonic()
for suffix,raw in [('stdout.private.json',r.stdout),('stderr.private.txt',r.stderr)]:
 with (D/('preflight-'+tag+'.'+suffix)).open('xb') as f:f.write(raw)
receipt=dict(**intent,transport_rc=r.returncode,ended_monotonic=ended,RTT_seconds=ended-started,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/('preflight-'+tag+'-result.json')).open('x') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),phase=v.get('phase'),state=v.get('state'),compute_app_records=v.get('compute_app_records'),GPU_launched=False)))
