"""Immutable upload and one controller start; payload is injected last."""
from pathlib import Path
import base64,hashlib,json,re,sys
D=Path(__file__).resolve().parent;commit=sys.argv[1];assert re.fullmatch('[0-9a-f]{40}',commit);mf=D/'GPU_SOURCE_MANIFEST.json';pin=hashlib.sha256(mf.read_bytes()).hexdigest();m=json.loads(mf.read_bytes());files={n:base64.b64encode((D/n).read_bytes()).decode()for n in [*m['files_SHA256'],'GPU_SOURCE_MANIFEST.json']}
s='''from pathlib import Path
import base64,hashlib,json,subprocess,sys,os,datetime
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/Deep-fixed-Gemm-Cast-graph-r399-gpu-v2';assert not D.exists()and not(R/'results/Deep399-GPU-watch-v2.json').exists();D.mkdir()
for n,b in __R399_FINAL_PAYLOAD__.items():
 with(D/n).open('xb')as f:f.write(base64.b64decode(b))
assert hashlib.sha256((D/'GPU_SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__R399_FINAL_PIN__
go=dict(execute_once=True,publication_commit=__R399_FINAL_COMMIT__,GPU_source_manifest_SHA256=__R399_FINAL_PIN__);p=R/'results/Deep399-GPU-execution-GO-v2.json'
with p.open('x')as f:json.dump(go,f,indent=2);f.flush();os.fsync(f.fileno())
sys.path.insert(0,str(D));import controller;controller.verify(R,go)
args=['python3','-B',str(D/'controller.py'),str(p)]
with(R/'logs/Deep399-GPU-controller-v2.log').open('x')as log:proc=subprocess.Popen(args,env=controller.clean_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),publication_commit=__R399_FINAL_COMMIT__,GPU_source_manifest_SHA256=__R399_FINAL_PIN__,controller_PID=proc.pid,owned_original_PGID=proc.pid,argv=args,initial_exit=proc.poll(),GPU_submitted_by_this_helper=False,no_retry=True)
with(R/'results/Deep399-GPU-controller-submitted-v2.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
'''
s=s.replace('__R399_FINAL_PIN__',repr(pin)).replace('__R399_FINAL_COMMIT__',repr(commit)).replace('__R399_FINAL_PAYLOAD__',repr(files));compile(s,'remote399-final-submit.py','exec')
for n,h in m['files_SHA256'].items():assert hashlib.sha256(base64.b64decode(files[n])).hexdigest()==h
p=D.parents[4]/'Deep399-final-GPU-published-submit-v2.py';p.write_bytes(s.encode());print(p)
