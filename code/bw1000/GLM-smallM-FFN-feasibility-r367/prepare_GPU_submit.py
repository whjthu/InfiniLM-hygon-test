"""Upload immutable GPU additions and start the published controller once."""
from pathlib import Path
import sys,json,hashlib,base64,re
D=Path(__file__).resolve().parent;commit=sys.argv[1];assert re.fullmatch('[0-9a-f]{40}',commit)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();mf=D/'GPU_SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes());files={n:base64.b64encode((D/n).read_bytes()).decode()for n in [*m['files_SHA256'],'GPU_SOURCE_MANIFEST.json']}
script='''from pathlib import Path
import json,hashlib,base64,subprocess,os,datetime
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/GLM-smallM-FFN-feasibility-r367';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert D.is_dir()and not(R/'results/GLM367-watch-r1.json').exists()
for n,b in FILE_PAYLOAD.items():
 p=D/n;data=base64.b64decode(b);assert not p.exists(),str(p)
 with p.open('xb')as f:f.write(data)
assert sha(D/'GPU_SOURCE_MANIFEST.json')==MANIFEST_PIN
go=dict(execute_once=True,publication_commit=SOURCE_COMMIT,GPU_source_manifest_SHA256=MANIFEST_PIN)
path=R/'results/GLM367-execution-GO.json'
with path.open('x')as f:json.dump(go,f,indent=2)
import sys
sys.path.insert(0,str(D));import controller
controller.verify(R,go)
args=['python3','-B',str(D/'controller.py'),'--execution-go',str(path)]
with(R/'logs/GLM367-controller-r1.log').open('x')as log:p=subprocess.Popen(args,env=controller.clean_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),publication_commit=SOURCE_COMMIT,GPU_source_manifest_SHA256=MANIFEST_PIN,controller_PID=p.pid,owned_controller_PGID=p.pid,argv=args,execution_GO_path=str(path),execution_GO_SHA256=sha(path),initial_exit=p.poll(),GPU_submitted_by_this_script=False,controller_responsibility='Onefreshn03/oneexactGPUstep<=600s, Slurm+srun propagation and PID/KFD post, no replay',original_CPU='910407.726')
with(R/'results/GLM367-controller-submitted-r1.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
'''
for k,v in [('FILE_PAYLOAD',files),('MANIFEST_PIN',sha(mf)),('SOURCE_COMMIT',commit)]:script=script.replace(k,repr(v))
compile(script,'remote367-GPU-submit.py','exec');p=D.parents[3]/'remote367-GPU-submit.py';p.write_bytes(script.encode());print(p)
