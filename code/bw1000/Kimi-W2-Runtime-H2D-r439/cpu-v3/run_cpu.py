"""Compile one thin HOST caller and link two executable arms. Do not run them."""
from pathlib import Path
import hashlib,json,os,re,subprocess,time,traceback
D=Path(__file__).resolve().parent
B=Path('/refactor/candidates/Kimi-W2-Runtime-H2D-r439-CPU-v3')
C=Path('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM')
started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,HOST_compile=0,Runtime_relink=0,executable_links=0,Runtime_executions=0,GPU=0,Native=False,commands=[])
def run(argv,name,limit=40):
 left=255-(time.monotonic()-started);assert left>0
 t=time.monotonic();p=subprocess.run(argv,cwd=C,capture_output=True,text=True,timeout=min(limit,left))
 (B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr)
 q['commands'].append(dict(name=name,argv=argv,rc=p.returncode,wall_s=time.monotonic()-t,stdout_SHA256=sha(B/(name+'.stdout')),stderr_SHA256=sha(B/(name+'.stderr'))))
 assert p.returncode==0,dict(operation=name,rc=p.returncode,stderr=p.stderr[-4000:])
 return p.stdout
def deps(path):
 s=path.read_text().replace('\\\n',' ')
 return {str((C/p).resolve()if not p.startswith('/')else Path(p).resolve())for p in s.split(':',1)[1].split()}
def main():
 B.mkdir(exist_ok=False)
 try:
  q['actual_CPU']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
  q['GPU_visibility']={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']};assert set(q['GPU_visibility'].values())=={'-1'}
  q['source_manifest_SHA256']=sha(D/'SOURCE_MANIFEST.json');assert q['source_manifest_SHA256']==os.environ['R439_SOURCE_SHA256']
  for name,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/name)==h,name
  d=json.loads((D/'DERIVATION.json').read_bytes())
  for p,h in d['prior_files_SHA256'].items():assert sha(p)==h,p
  old=Path('/refactor/incoming/Kimi-loader-row-copy-r436/runtime-v4');c=json.loads((old/'BUILD_CONTRACT.json').read_bytes())
  receipt=json.loads(Path(d['original_884_receipt']).read_bytes());assert receipt['actual_CPU']=='910407.884'and not receipt['passed']
  for item in receipt['new_HOST_compiled']:
   assert sha(item['object'])==item['object_SHA256']
   for p,h in item['dependency_SHA256'].items():assert sha(p)==h,p
  for p,h in c['runtime_objects_SHA256'].items():assert sha(p)==h,p
  q['Runtime_providers']=d['Runtime_providers']
  for v in q['Runtime_providers'].values():assert sha(v['path'])==v['SHA256']
  flags=c['HOST_prefix']+['-pthread','-Werror=multichar']
  obj=B/'caller.o';pre=B/'caller.pre.d';md=B/'caller.d'
  run(flags+['-M','-MF',str(pre),'-MT',str(obj),str(D/'caller.cpp')],'caller-pre-MD',35)
  pins={p:sha(p)for p in deps(pre)}
  run(flags+['-MD','-MF',str(md),'-c',str(D/'caller.cpp'),'-o',str(obj)],'caller-HOST-compile',110);q['HOST_compile']=1
  assert deps(md)==set(pins)and all(sha(p)==h for p,h in pins.items())
  q['compiled']=dict(source=str(D/'caller.cpp'),source_SHA256=sha(D/'caller.cpp'),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependencies_SHA256=pins)
  sdk='/refactor/install/hygon-kimi-5f91ad7-r3672/lib'
  libs=['-L'+sdk,'-linfiniops','-linfiniccl','-linfinirt','-Wl,-rpath,'+sdk,'-L/opt/dtk/cuda/cuda/lib64','-lcudart','-ldl']
  q['executables']={}
  for label,v in q['Runtime_providers'].items():
   ldd=run(['ldd','-r',v['path']],label+'-Runtime-ldd',25);assert 'not found'not in ldd and 'undefined symbol'not in ldd
   exe=B/label;run(['/usr/bin/g++','-m64','-pthread',str(obj),v['path'],*libs,'-o',str(exe)],label+'-caller-link',25);q['executable_links']+=1
   dynamic=run(['readelf','-d',str(exe)],label+'-DT-NEEDED',15);assert v['path']in dynamic
   other=[w['path']for k,w in q['Runtime_providers'].items()if k!=label];assert all(p not in dynamic for p in other)
   check=run(['ldd','-r',str(exe)],label+'-caller-ldd',25);assert 'not found'not in check and 'undefined symbol'not in check
   q['executables'][label]=dict(path=str(exe),SHA256=sha(exe),Runtime=v)
  q.update(passed=True,device_execution_passed=False,Model_GO=False,shared_install_modified=False)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  q.update(completed=True,wall_s=time.monotonic()-started);(B/'result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','HOST_compile','executable_links','wall_s']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
