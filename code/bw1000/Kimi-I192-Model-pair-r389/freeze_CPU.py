"""Freeze only this private source and generate one source-bound CPU SSH payload."""
from pathlib import Path
import ast,base64,hashlib,json

D=Path(__file__).resolve().parent
G=D.parents[2];L=G.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8',newline='\n')
for folder in [D/'target',D/'controller']:
    files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
    manifest=dict(files_SHA256=files,GPU_GO=False,Native_SHA256='4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505',worker_manifest_SHA256='61ac8b04af464d76b4ce46f0793e6d8a2a8e57f41224d6228c5b826bda6ffd31')
    if folder.name=='controller':manifest['outer_wrapper_SHA256']=files['outer.sh']
    dump(folder/'SOURCE_MANIFEST.json',manifest)
files={p.relative_to(D).as_posix():sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p!=D/'SOURCE_MANIFEST.json' and '__pycache__'not in p.parts}
manifest=dict(files_SHA256=files,GPU_GO=False,CPU_claim=json.loads((D/'DESIGN.json').read_bytes())['CPU_claim'],scope='CPUwholeentry only; no Nativecompile/dlopen/weights/GPU')
dump(D/'SOURCE_MANIFEST.json',manifest)
h=sha(D/'SOURCE_MANIFEST.json')
payload={p.relative_to(D).as_posix():base64.b64encode(p.read_bytes()).decode() for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__'not in p.parts}
script='''from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
MF=__MANIFEST_SHA__
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-I192-Model-pair-r389'
label='Kimi-I192-r389-CPU'
raw=subprocess.check_output(['sacct','-S','2026-10-31','-j','910407','-n','-P','-o','JobIDRaw,JobName%160,State'],text=True,timeout=25)
assert not any(line.split('|')[1]==label for line in raw.splitlines() if '|'in line),'Original label exists; reconcile, never resubmit'
assert not (R/'candidates/Kimi-I192-Model-pair-r389').exists()
assert not D.exists();D.mkdir()
for name,value in PAYLOAD.items():
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value))
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==MF
for p in D.rglob('*.py'):ast.parse(p.read_text())
intent=R/'results/Kimi-I192-r389-CPU-intent.json'
with intent.open('x')as f:json.dump(dict(source_manifest_SHA256=MF,CPU_only=True,GPU=0,retry=False),f)
env=dict(os.environ,WAVE_INSIDE_R2323=D.name+'/CPU_inside.sh',KIMI_R389_CPU_SOURCE_SHA=MF)
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:05:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(R/'logs'/(label+'-client.log')).open('x')as log:
 p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(submitted=True,CPU_only=True,GPU=0,PID=p.pid,label=label,source_manifest_SHA256=MF)))
'''
script=script.replace('__MANIFEST_SHA__',repr(h)).replace('PAYLOAD.items()',repr(payload)+'.items()')
ast.parse(script)
(L/'remote389-upload-CPU.py').write_text(script,encoding='utf8',newline='\n')
print(json.dumps(dict(source_manifest_SHA256=h,members=len(files),helper_AST=True,GPU_GO=False)))
