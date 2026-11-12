"""Fresh read-only source/model/device/port admission for the published reference."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
base=claim+':code/bw1000/training-qwen-A100-first-updates-v2/'
values={name:json.loads(subprocess.check_output(git+['show',base+name])) for name in ('FRAMEWORK_FILES.json','MODEL_FILES.json','GPU_BINDING.json','INPUT_FILES.json','PRIOR_QUALIFICATIONS.json','INSTALLED_SOURCE_FILES.json')}
remote=('BOUND='+repr(values)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,importlib.metadata as metadata,json,os,shutil,socket,stat,subprocess
phase='namespace';result=None
try:
 W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-A100-first-updates-v2');assert not W.exists()
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 fm=BOUND['FRAMEWORK_FILES.json'];count=0;phase='parent-source'
 for name,pin in fm['files'].items():
  if name in fm['deltas']:continue
  p=Path(fm['parent_directory'])/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode'];count+=1
 assert count==337
 phase='prior-original-qualifications'
 for q in BOUND['PRIOR_QUALIFICATIONS.json']['qualifications']:
  for pin in q['files']:
   p=Path(q['root'])/pin['name'];assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='model-files';model=BOUND['MODEL_FILES.json'];model_root=Path(model['model_root'])
 for name,pin in model['files'].items():
  p=model_root/name;s=p.stat();assert p.is_file() and not p.is_symlink() and s.st_size==pin['bytes']
  if name.endswith('.safetensors'):assert dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)==model['weight_stats'][name]
  else:assert sha(p)==pin['SHA256']
 phase='original-inputs'
 for pin in BOUND['INPUT_FILES.json']['files']:
  p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='installed-native-source'
 for name,pin in BOUND['INSTALLED_SOURCE_FILES.json'].items():
  package={'transformers':'transformers','swift':'ms-swift','peft':'peft'}[name.split('/')[0]];p=Path(metadata.distribution(package).locate_file(name));assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='GPU-identity-and-idleness'
 def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=12).strip().splitlines()
 rows=[[v.strip() for v in line.split(',')] for line in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'])]
 binding=BOUND['GPU_BINDING.json'];assert len(rows)==8 and {r[0]:r[1] for r in rows}==binding['UUID_by_index'] and {r[0]:r[2] for r in rows}==binding['PCI_by_index']
 apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
 assert not apps and all(r[3]=='NVIDIA A100-SXM4-40GB' and int(r[5])<=16 and int(r[6])==0 for r in rows)
 phase='port-and-disk'
 with socket.socket() as s:s.bind(('127.0.0.1',29895))
 free=shutil.disk_usage(W.parent).free;assert free>=80*1024**3
 assert not any(k in os.environ for k in ('RANK','LOCAL_RANK','WORLD_SIZE','GROUP_RANK'))
 result=dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_absent=True,parent_source_files=337,model_files=14,weight_full_SHA_reused_with_exact_stats=4,
  original_input_files=67,installed_native_source_files=len(BOUND['INSTALLED_SOURCE_FILES.json']),GPU_cards=rows,compute_apps=apps,master_port_bindable=True,free_disk_bytes=free,
  versions={k:metadata.version(k) for k in ['torch','torchvision','transformers','ms-swift','peft','accelerate','Pillow','tokenizers']},GPU_launched=False,Torch_imported=False)
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,GPU_launched=False,Torch_imported=False)
print(json.dumps(result))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=90)
with (D/'preflight.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'preflight.stderr.private.txt').open('xb') as f:f.write(r.stderr)
value=dict(claim=claim,transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:value['value']=json.loads(r.stdout)
with (D/'preflight-result.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
v=value.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),phase=v.get('phase'),error_type=v.get('error_type'),GPU_cards=len(v.get('GPU_cards',[])),compute_apps=len(v.get('compute_apps',[])),master_port_bindable=v.get('master_port_bindable'),GPU_launched=False)))
