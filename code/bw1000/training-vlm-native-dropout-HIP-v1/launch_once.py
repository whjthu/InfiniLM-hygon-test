"""Run only the exact published source and A100 reference; preserve unknown outcomes."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;G=Path(sys.argv[1]);claim=sys.argv[2]
def blob(name):return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':'+name])
task=json.loads(blob('coordination/tasks/training-vlm-native-dropout-HIP-v1.json'));sources={}
for name,pin in task['source_SHA256'].items():
 raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin,name;sources[name]=raw
raw=(BASE/'A100-native-reference.pt').read_bytes();reference=task['A100_reference'];assert len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['SHA256']
bound={n:{'b64':base64.b64encode(sources[n]).decode(),'SHA256':task['source_SHA256'][n]} for n in task['remote_files']}
ref={'b64':base64.b64encode(raw).decode(),'SHA256':reference['SHA256']}
prelude=(BASE.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(Path(__file__).resolve())};exec(compile(prelude.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
with (BASE/'launch-intent.json').open('x') as f:json.dump({'claim_commit':claim,'node':task['node'],'reference_SHA256':reference['SHA256']},f)
payload=(f'CLAIM={claim!r}\nFILES={bound!r}\nREFERENCE={ref!r}\n'+sources['receive_once.py'].decode()).encode()
r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=400)
with (BASE/'original.stdout').open('xb') as f:f.write(r.stdout)
with (BASE/'original.stderr').open('xb') as f:f.write(r.stderr)
out={'transport_rc':r.returncode,'stdout_SHA256':hashlib.sha256(r.stdout).hexdigest(),'stderr_SHA256':hashlib.sha256(r.stderr).hexdigest()}
if r.returncode==0:out['value']=json.loads(r.stdout)
with (BASE/'transport-outcome.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
