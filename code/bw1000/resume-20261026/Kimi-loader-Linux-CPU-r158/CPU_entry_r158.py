"""Bounded standalone CPU entry; never Slurm/Git/SSH/GPU/torch/model payload."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import time

D=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--compiler',default='g++');p.add_argument('--smoke',action='store_true');a=p.parse_args()
out=pathlib.Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();record={'CPU_only':True,'GPU':False,'Native_dlopen':False,'model_payload_read':False,'platform':platform.platform(),'host':platform.node(),'SLURM_JOB_ID':os.environ.get('SLURM_JOB_ID'),'SLURM_CPUS_PER_TASK':os.environ.get('SLURM_CPUS_PER_TASK'),'commands':[],'passed':False,'source_SHA256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in D.iterdir()if f.suffix in('.cpp','.hpp','.py','.sh','.diff')}}
flags=['-std=c++17','-O3','-pthread','-fno-lto'];record['compile_flags_all_TUs']=flags
record['CPU_affinity_ids']=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
def run(cmd):
 remaining=170-(time.monotonic()-start)
 if remaining<=0:raise TimeoutError('170-second whole CPU entry budget exhausted')
 t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=min(remaining,60));record['commands'].append({'argv':cmd,'rc':r.returncode,'wall_s':time.monotonic()-t,'stderr':r.stderr[-4096:]})
 if r.returncode:raise RuntimeError('CPU command failed: '+r.stderr[-1024:])
 return r.stdout
try:
 record['compiler_version']=run([a.compiler,'--version']).splitlines()[0]
 objects=[]
 for source in('baseline_r158.cpp','candidate_r158.cpp','driver_r158.cpp'):
  obj=out/(pathlib.Path(source).stem+'.o');run([a.compiler,*flags,'-I',str(D),'-c',str(D/source),'-o',str(obj)]);objects.append(str(obj))
 exe=out/('bench.exe'if os.name=='nt'else'bench');run([a.compiler,*flags,*objects,'-o',str(exe)])
 record['benchmark']=json.loads(run([str(exe)]+(['--smoke']if a.smoke else[])))
 for src in('regression_r157.cpp','exact_geometry_contract_r157.cpp'):
  ex=out/(pathlib.Path(src).stem+('.exe'if os.name=='nt'else''));run([a.compiler,*flags,'-I',str(D),str(D/src),'-o',str(ex)]);record[pathlib.Path(src).stem]=json.loads(run([str(ex)]))
 record['passed']=True
finally:
 record['whole_entry_wall_s']=time.monotonic()-start
 try:
  import resource
  record['child_max_RSS_KiB_Linux']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
 except ImportError:record['child_max_RSS_KiB_Linux']=None
 (out/'CPU-result-r158.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'passed':record['passed'],'result':str(out/'CPU-result-r158.json'),'whole_entry_wall_s':record['whole_entry_wall_s']}))
