import base64,hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;G=Path(sys.argv[1]);claim=sys.argv[2]
def blob(name):return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':'+name])
task=json.loads(blob('coordination/tasks/training-vlm-lora-dropout-session-CPU-v1.json'));source={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;source[name]=raw
bound={n:{'b64':base64.b64encode(source[n]).decode(),'SHA256':task['source_SHA256'][n]} for n in task['remote_files']}
with (BASE/'launch-intent.json').open('x') as f:json.dump({'claim_commit':claim,'GPU_allocation':0},f)
config=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(config),'a100-26','python3 -B -u -'],input=(f'CLAIM={claim!r}\nFILES={bound!r}\n'+source['receive_once.py'].decode()).encode(),capture_output=True,timeout=220)
with (BASE/'original.stdout').open('xb') as f:f.write(r.stdout)
with (BASE/'original.stderr').open('xb') as f:f.write(r.stderr)
v={'transport_rc':r.returncode,'stdout_SHA256':hashlib.sha256(r.stdout).hexdigest(),'stderr_SHA256':hashlib.sha256(r.stderr).hexdigest()}
if r.returncode==0:v['value']=json.loads(r.stdout)
with (BASE/'transport-outcome.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
