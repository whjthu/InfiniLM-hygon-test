"""One bounded CPU-only compile/semantic/paired copy card on compute node."""
import hashlib,json,os,pathlib,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/candidates/Kimi-loader-row-copy-r436-CPU')
start=time.monotonic();B.mkdir(exist_ok=False)
q={'completed':False,'passed':False,'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
   'scope':'CPU standalone only, no framework/Native/weights/GPU','new_GPU':0,'commands':[]}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(cmd,name):
    left=150-(time.monotonic()-start);assert left>0
    q['commands'].append(cmd)
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=min(60,left))
    (B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr)
    assert p.returncode==0,{'operation':name,'rc':p.returncode,'stderr':p.stderr[-3000:]}
    return p.stdout
try:
    assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());q['source_manifest_SHA256']=sha(D/'SOURCE_MANIFEST.json')
    assert q['source_manifest_SHA256']==os.environ['R436_SOURCE_SHA256']
    for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
    assert sha(D/'cpu_strided_copy_r2515.hpp')=='0381c2635110bc40c99ccf026a1dc678aad934ddca34aa571e844a3f942b38dc'
    q['Slurm_entry_affinity']=sorted(os.sched_getaffinity(0));assert q['Slurm_entry_affinity']
    os.sched_setaffinity(0,{q['Slurm_entry_affinity'][0]})
    q['actual_cpu_affinity']=sorted(os.sched_getaffinity(0));assert len(q['actual_cpu_affinity'])==1
    q['compiler']=run(['/usr/bin/g++','--version'],'compiler').splitlines()[0]
    for source,name in [('check_copy.cpp','check'),('bench_copy.cpp','bench')]:
        run(['/usr/bin/g++','-std=c++17','-O3','-Wall','-Wextra','-Werror',str(D/source),'-o',str(B/name)],name+'.compile')
    q['semantic']=json.loads(run([str(B/'check')],'semantic'))
    q['paired']=json.loads(run([str(B/'bench')],'paired'))
    assert q['semantic']['passed']and q['paired']['passed']
    maps=pathlib.Path('/proc/self/maps').read_text()
    q['project_Native_maps']=[s for s in maps.splitlines()if any(v in s for v in ['libKimi','libGLM','libDeep','libinfinilm','libinfinirt','_infinilm'])]
    assert not q['project_Native_maps']
    q['passed']=True
except BaseException as error:
    q['error']=repr(error);q['traceback']=traceback.format_exc();raise
finally:
    q['completed']=True;q['seconds']=time.monotonic()-start
    (B/'result.json').write_text(json.dumps(q,indent=2)+'\n')
