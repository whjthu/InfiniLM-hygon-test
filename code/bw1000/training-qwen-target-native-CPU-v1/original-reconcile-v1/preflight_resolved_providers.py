"""Read exact target runtime/parent files before any new CPU installation."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;B=D.parent
source=D/'publication/code/bw1000/training-qwen-target-native-CPU-v1'
bound={n:json.loads((source/n).read_bytes()) for n in ('FRAMEWORK_FILES.json','RUNTIME_SOURCE_PINS.json')}
remote=('BOUND='+repr(bound)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,stat,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-qwen-target-native-CPU-v1'
phase='namespace';mismatches=[]
try:
 assert not W.exists()
 def command(a):return subprocess.check_output(a,text=True,timeout=12).strip()
 assert command(['squeue','-h','-j','1087656','-o','%T|%D'])=='RUNNING|40'
 assert not any('|training-qwen-target-native-CPU-v1|' in line for line in command(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines())
 phase='runtime-providers'
 for pin in BOUND['RUNTIME_SOURCE_PINS.json']['files'].values():
  name=pin['path']
  if name.startswith('/minicpm-transformers/'):p=R/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site'/name.removeprefix('/minicpm-transformers/')
  elif name.startswith('/usr/local/lib/python3.10/dist-packages/peft/'):
   p=R/'containers/dtk26-sandbox-r1'/name.lstrip('/')
  else:
   assert name.startswith('/swift-runtime/site/');p=R/'results/training-vlm-swift-offline-runtime-v1/site'/name.removeprefix('/swift-runtime/site/')
  if not p.is_file():mismatches.append(dict(path=name,exists=False));continue
  raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest()
  if len(raw)!=pin['bytes'] or actual!=pin['SHA256']:mismatches.append(dict(path=name,exists=True,bytes=len(raw),SHA256=actual))
 assert not mismatches
 phase='immutable-parent-source';fm=BOUND['FRAMEWORK_FILES.json'];count=0
 for name,pin in fm['files'].items():
  if name in fm['deltas']:continue
  p=Path(fm['parent_source'])/name;assert p.stat().st_size==pin['bytes'] and stat.S_IMODE(p.stat().st_mode)==pin['mode'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'];count+=1
 assert count==342-len(fm['deltas'])
 result=dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_absent=True,parent_running_nodes=40,runtime_files=28,unchanged_parent_files=count,GPU_cards=0,Torch_imported=False,new_Slurm_steps=0)
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,mismatches=mismatches,GPU_cards=0,Torch_imported=False,new_Slurm_steps=0)
print(json.dumps(result))
'''
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=75)
for name,raw in [('preflight3.stdout.private.json',r.stdout),('preflight3.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
result=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/'preflight3-result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),phase=v.get('phase'),runtime_mismatches=len(v.get('mismatches',[])),unchanged_parent_files=v.get('unchanged_parent_files'),new_Slurm_steps=0,GPU_cards=0)))
