"""Pure stdlib CPU-v2 contract verifier, also executed by the future HOST launcher."""
from pathlib import Path
import ast
import hashlib
import json
import math
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify(d=None):
 d=Path(d or Path(__file__).parent).resolve();m=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());c=json.loads((d/'CPU_RESOURCE_CLAIM.json').read_bytes());p=json.loads((d/'BUILD_PLAN.json').read_bytes())
 assert all(H(d/n)==h for n,h in m['files_SHA256'].items())
 assert H(d/'SOURCE_MANIFEST.json')==c['source_manifest_SHA256']
 assert c['label']=='Deep-C4-causal-consumer-r483-CPU-v2' and p['candidate']=='/refactor/candidates/'+c['label']
 assert (c['CPU'],c['memory_MiB'],c['GPU'],c['new_HOST_TUs'],c['new_GPU_translation_units'])==(4,8192,0,1,0)
 assert (c['Slurm_seconds'],c['inside_seconds'],c['work_seconds'])==(300,270,265)
 assert c['CPU_GO'] and c['Native_compile_GO'] and not c['GPU_GO'] and not c['Model_GO']
 sm=json.loads((d/'native-src/SOURCE_MANIFEST.json').read_bytes());assert H(d/'native-src/SOURCE_MANIFEST.json')=='bd572de4ad023a69d1fd6398dd09cf723f3c3a4f465c389a2d7b10144a94cb2d'
 assert all(H(d/'native-src'/n)==h for n,h in sm['files_SHA256'].items())
 assert len(p['units'])==1 and p['units'][0]['kind']=='HOST' and not p['objects_SHA256']
 assert p['units'][0]['required_local_dependencies']==['native-src/input_guard.hpp']
 assert len(p['links'])==1 and p['links'][0]['object_count']==1 and len(p['links'][0]['required_exports'])==13
 assert p['class_header_SHA256']=='38b53b223be516315c8e6479e20a9cfdcdf089f0a1e59fe328faf2492d4991b1'
 assert any(r['SHA256']=='392ec4350bc150ba042b94dbb5732183c26cd8418a533e45d8fb56eecdf04242' for r in p['references'])
 build=(d/'build_plan.py').read_text();inside=(d/'CPU_inside.sh').read_text();launch=(d/'launch_CPU.py').read_text()
 assert "registered=str(D/relative);assert registered in names" in build
 assert build.index("assert not unmatched")<build.index("+'-compile'")
 assert "work_remaining(start,deadline,time.monotonic(),time.time())" in build
 assert 'TUNING_CPU_EPOCH=' in launch and 'TUNING_CPU_DEADLINE=' in launch and 'epoch=time.time();deadline=epoch+300' in launch
 assert 'verify_package.py' in launch and "epoch=epoch,deadline=deadline" in launch
 assert 'float(os.environ["TUNING_CPU_DEADLINE"])-time.time()-7.0' in inside and '"${remaining}s"' in inside and '--kill-after=5s' in inside
 for name in m['files_SHA256']:
  if name.endswith('.py'):ast.parse((d/name).read_bytes(),filename=name)
 tree=ast.parse(build);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='work_remaining')
 namespace={'math':math};exec(compile(ast.Module(body=[f],type_ignores=[]),'<actual work_remaining>','exec'),namespace)
 work=namespace['work_remaining'];assert work(100,1300,100,1000)==265
 assert work(100,1300,101,1100)==193 and work(100,1300,110,1290)==3
 rejected=0
 for a in [(100,1300,99,1000),(100,1300,364,1000),(100,1300,100,1291),(100,1300,100,1300),
           (100,float('nan'),100,1000),(100,1300,float('inf'),1000),(float('nan'),1300,100,1000),(100,1300,100,float('nan'))]:
  try:work(*a)
  except (ValueError,TimeoutError):rejected+=1
  else:raise AssertionError('invalid deadline/clock accepted')
 assert rejected==8
 return dict(passed=True,actual_budget_function_valid_cases=3,rejected_expired_or_invalid_clocks=8,
             required_new_header_registered=True,all_fresh_M_before_compile=True,single_original_epoch_propagated=True,
             target_compile=False,Native=False,SDK=False,GPU=False)

if __name__=='__main__':print(json.dumps(verify(),indent=2))
