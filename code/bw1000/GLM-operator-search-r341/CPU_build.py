import pathlib,json,hashlib,subprocess,time,os
inc=pathlib.Path(__file__).parent;b=pathlib.Path('/refactor/candidates/GLM-operator-search-r341');b.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((inc/'SOURCE_MANIFEST.json').read_text());assert all(sha(inc/n)==h for n,h in m['files_SHA256'].items())
q={'CPU_only':True,'GPU_initialized':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_manifest_SHA256':sha(inc/'SOURCE_MANIFEST.json')};t=time.monotonic()
a=['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-shared','-Xcompiler=-fPIC',str(inc/'fused.cu'),'-o',str(b/'fused.so')]
z=subprocess.run(a,capture_output=True,text=True,timeout=420);q.update(command=a,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr,seconds=time.monotonic()-t)
if z.returncode==0:
 q['library_SHA256']=sha(b/'fused.so');z=subprocess.run(['ldd',str(b/'fused.so')],capture_output=True,text=True);q['ldd']=z.stdout;assert 'not found' not in z.stdout
(b/'CPU_RESULT.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['returncode']==0
