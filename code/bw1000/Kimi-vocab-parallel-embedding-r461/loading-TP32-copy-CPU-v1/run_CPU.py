from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
D=Path(__file__).resolve().parent;B=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/candidates/Kimi-loader-TP32-copy-CPU-v1');start=time.monotonic();B.mkdir(exist_ok=False)
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Native_dlopen=False,SDK_execution=False,weights=False,commands=[],actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
def save():(B/'result.json').write_text(json.dumps(q,indent=2)+'\n')
def run(cmd,name):
 left=45-(time.monotonic()-start);assert left>0;q['phase']=name;q['commands'].append(cmd);save()
 z=subprocess.run(cmd,capture_output=True,text=True,timeout=left)
 for kind,text in [('stdout',z.stdout),('stderr',z.stderr)]:p=B/(name+'.'+kind);p.write_text(text);q.setdefault('logs',{})[p.name]=dict(bytes=p.stat().st_size,SHA256=H(p))
 assert z.returncode==0,(name,z.returncode,z.stderr[-2000:]);return z.stdout
try:
 assert os.uname().nodename.split('.')[0]=='a14r2n20'
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert not os.environ.get('LD_PRELOAD')
 q['source_manifest_SHA256']=H(D/'SOURCE_MANIFEST.json');assert q['source_manifest_SHA256']==os.environ['KIMI_LOADER_CPU_MF']
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in m['files_SHA256'].items():assert H(D/name)==h,name
 q['entry_mask']=sorted(os.sched_getaffinity(0));assert q['entry_mask'];os.sched_setaffinity(0,{q['entry_mask'][0]});q['actual_mask']=sorted(os.sched_getaffinity(0));assert len(q['actual_mask'])==1
 q['compiler']=run(['/usr/bin/g++','--version'],'compiler').splitlines()[0]
 run(['/usr/bin/g++','-std=c++17','-O3','-Wall','-Wextra','-Werror',str(D/'caller.cpp'),'-o',str(B/'caller')],'compile')
 q['caller_SHA256']=H(B/'caller');q['ELF']=run(['readelf','-d',str(B/'caller')],'ELF')
 assert not any(s in q['ELF']for s in ['libinfinilm','libinfiniops','libinfinirt','libcuda','libcudart'])
 q['consumer']=json.loads(run([str(B/'caller')],'consumer'));assert q['consumer']['passed']
 q['project_maps']=[x for x in Path('/proc/self/maps').read_text().splitlines()if any(s in x for s in ['libKimi','libGLM','libDeep','libinfinilm','libinfiniops','libcudart'])];assert not q['project_maps']
 q['passed']=True
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
finally:q.update(completed=True,elapsed_s=time.monotonic()-start);save();print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','error','elapsed_s']}))
raise SystemExit(0 if q['passed']else 1)
