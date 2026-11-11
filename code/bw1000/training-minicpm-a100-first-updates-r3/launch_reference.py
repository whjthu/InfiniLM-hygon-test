"""Invoke only the frozen published eight-card diagnostic reference, once."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;G=Path(sys.argv[1]);CLAIM=sys.argv[2]
def blob(path):return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',CLAIM+':'+path])
task=json.loads(blob('coordination/tasks/training-minicpm-a100-first-updates-r3.json'))
sources={}
for n,pin in task['source_SHA256'].items():
    raw=blob(task['source']+'/'+n);assert hashlib.sha256(raw).hexdigest()==pin,n
    sources[n]=raw
files={n:{'b64':base64.b64encode(sources[n]).decode(),'SHA256':task['source_SHA256'][n]} for n in task['remote_files']}
code=f'CLAIM={CLAIM!r}\nFILES={files!r}\n'.encode()+sources['receive_reference.py']
config=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
with (BASE/'launch-intent.json').open('x') as f:json.dump({'claim_commit':CLAIM,'code_SHA256':hashlib.sha256(code).hexdigest()},f)
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(config),'a100-26','python3 -B -u -'],input=code,capture_output=True,timeout=1080)
with (BASE/'original.stdout').open('xb') as f:f.write(r.stdout)
with (BASE/'original.stderr').open('xb') as f:f.write(r.stderr)
out={'transport_rc':r.returncode,'stdout_SHA256':hashlib.sha256(r.stdout).hexdigest(),'stderr_SHA256':hashlib.sha256(r.stderr).hexdigest()}
if r.returncode==0:out['value']=json.loads(r.stdout)
with (BASE/'transport-outcome.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
