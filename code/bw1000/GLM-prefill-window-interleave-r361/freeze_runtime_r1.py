from pathlib import Path
import ast,json,hashlib,base64
D=Path(__file__).resolve().parent;OUT=D/'runtime-r1';BASE=D.parent/'GLM-two-cohort-Model-r355';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((BASE/'owner/SOURCE_MANIFEST.json').read_bytes());files={p.relative_to(OUT/'owner').as_posix():sha(p)for p in sorted((OUT/'owner').rglob('*'))if p.is_file()and p.name!='SOURCE_MANIFEST.json'}
changed=sorted(n for n,h in old['files_SHA256'].items()if files[n]!=h)
assert set(changed)=={'owner_r290.py','model_entry_r290.py','GPU_inside_r290.sh','client_r355.py','workload_r355.py','PREPARATION.json'},changed
assert set(files)==set(old['files_SHA256'])|{'prefill_interleave.py'}
ownerMF=dict(schema=1,Native_SHA256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',GPU_GO=False,base_355_owner_manifest_SHA256='32f24ae36f4148c9eac9f86294a4bcccb3c1f269d6b1123cb9750d88a0693366',interleave_snapshot_manifest_SHA256='11cf0416e344e0fe8795c809c2cf54a87588f9dc6c8093a8d203749068deed8e',changed_from_355=changed,files_SHA256=files)
(OUT/'owner/SOURCE_MANIFEST.json').write_bytes((json.dumps(ownerMF,indent=2)+'\n').encode())
files={p.relative_to(OUT/'controller').as_posix():sha(p)for p in sorted((OUT/'controller').rglob('*'))if p.is_file()and p.name!='SOURCE_MANIFEST.json'};assert len(files)==7
(OUT/'controller/SOURCE_MANIFEST.json').write_bytes((json.dumps(dict(schema=1,Native_SHA256=ownerMF['Native_SHA256'],GPU_GO=False,base_355_controller_manifest_SHA256=sha(BASE/'controller/SOURCE_MANIFEST.json'),files_SHA256=files),indent=2)+'\n').encode())
for p in OUT.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
for p in OUT.rglob('*.sh'):assert b'\r'not in p.read_bytes(),str(p)
g=json.loads((OUT/'execution-GO.template.json').read_bytes());g.update(owner_manifest_SHA256=sha(OUT/'owner/SOURCE_MANIFEST.json'),controller_manifest_SHA256=sha(OUT/'controller/SOURCE_MANIFEST.json'));(OUT/'execution-GO.template.json').write_bytes((json.dumps(g,indent=2)+'\n').encode())
files={p.relative_to(OUT).as_posix():sha(p)for p in sorted(OUT.rglob('*'))if p.is_file()and p.name not in ['CPU_PACKAGE_MANIFEST.json','controller-CPU-local.json']}
(OUT/'CPU_PACKAGE_MANIFEST.json').write_bytes((json.dumps(dict(schema=1,GPU=0,scope='Actual owner/callback/client and32factory/cleanup at recording Native boundary; includes cancellation/partial-error failstop; not Model/Native numeric qualification',files_SHA256=files),indent=2)+'\n').encode())
data={p.relative_to(OUT).as_posix():base64.b64encode(p.read_bytes()).decode()for p in OUT.rglob('*')if p.is_file()and p.name!='controller-CPU-local.json'}
remote='''from pathlib import Path
import json,base64,hashlib,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-prefill-window-interleave-r361/runtime-r1';assert not D.exists();D.mkdir(parents=True)
files=FILES
for n,b in files.items():
 p=D/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:f.write(base64.b64decode(b))
m=json.loads((D/'CPU_PACKAGE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_'))};env['WAVE_INSIDE_R2323']='GLM-prefill-window-interleave-r361/runtime-r1/CPU_inside.sh'
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','--ntasks-per-node=1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=GLM-r361-whole-CPU-r1','--output='+str(R/'logs/GLM-r361-whole-CPU-r1-%J.log'),'--export=ALL','bash',str(D/'controller/resource_wave_host_alias_outer.sh'),'probe']
with(R/'logs/GLM-r361-whole-CPU-r1-client.log').open('x')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'GPU_submitted':False,'CPU_submitted':True,'CPU_PID':p.pid,'CPU_label':'GLM-r361-whole-CPU-r1','CPU_package_manifest_SHA256':hashlib.sha256((D/'CPU_PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),'owner_manifest_SHA256':hashlib.sha256((D/'owner/SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'controller_manifest_SHA256':hashlib.sha256((D/'controller/SOURCE_MANIFEST.json').read_bytes()).hexdigest()}))
'''.replace('files=FILES','files='+repr(data))
compile(remote,'remote-upload-CPU-r1','exec');(D/'remote_upload_CPU_r1.py').write_bytes(remote.encode())
print(json.dumps({'owner_manifest_SHA256':sha(OUT/'owner/SOURCE_MANIFEST.json'),'controller_manifest_SHA256':sha(OUT/'controller/SOURCE_MANIFEST.json'),'CPU_package_manifest_SHA256':sha(OUT/'CPU_PACKAGE_MANIFEST.json'),'GPU':0}))
