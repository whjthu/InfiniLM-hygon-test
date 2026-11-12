"""Close the installed runner/control-file hashes against the published CPU claim."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-qwen-A100-native-contract-CPU-v1.json']))
pins={remote:task['source_SHA256'][source] for remote,source in task['remote_files'].items()}
remote=('PINS='+repr(pins)+'\n').encode()+b'''
from pathlib import Path
import hashlib,json
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-native-update-contract-CPU-v1');checks={}
for name,pin in PINS.items():
 p=W/name;a=p.stat();raw=p.read_bytes();b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 assert hashlib.sha256(raw).hexdigest()==pin;checks[name]=dict(bytes=len(raw),SHA256=pin)
print(json.dumps(dict(passed=True,installed_files=len(checks),files=checks,read_only=True,Torch_imported=False,GPU_calls=0)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=45)
with (D/'closure.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'closure.stderr.private.txt').open('xb') as f:f.write(r.stderr)
value=dict(claim=claim,transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:value['value']=json.loads(r.stdout)
with (D/'installed-package-closure.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,passed=value.get('value',{}).get('passed'),installed_files=value.get('value',{}).get('installed_files'))))
