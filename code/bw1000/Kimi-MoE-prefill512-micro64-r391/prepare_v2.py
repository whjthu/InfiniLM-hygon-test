"""Two known fixes only; original published source/CPU782 remain immutable."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent;N=D/'v2';N.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((L/'Kimi391-original-782-closure.json').read_bytes());E=G/'results/bw1000/2026-11-02'
assert old['CPU-build.json']['SHA256']=='4c085f66411eb9b7449abb4506528313b11dd9db4fa7882f0f8443f302b9bfa6'
for name in ['CPU-build.json','container-ldd-r.log']:
 b=old[name]['text'].encode();assert hashlib.sha256(b).hexdigest()==old[name]['SHA256']
 if name=='CPU-build.json':(E/'Kimi391-CPU782-original-FAILED.json').write_bytes(b)
 else:(N/'original782-ldd-r.log').write_bytes(b)
for name in ['adapter.cpp','GPU_run.py','plan.py','check_CPU.py','CPU_build.py','CPU_inside.sh','invoke_CPU_once.py']:
 s=(D/name).read_text()
 if name=='adapter.cpp':
  needle='r.syncStream();double wall=std::chrono::duration<double,std::milli>(Clock::now()-tick).count();';assert s.count(needle)==1
  s=s.replace(needle,'held.clear();std::vector<rt::Tensor>().swap(held);r.syncStream();double wall=std::chrono::duration<double,std::milli>(Clock::now()-tick).count();')
 if name in ['CPU_build.py','CPU_inside.sh','invoke_CPU_once.py']:s=s.replace('Kimi-MoE-prefill512-micro64-r391','Kimi-MoE-prefill512-micro64-r391-v2')
 if name=='CPU_build.py':
  s=s.replace("actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']", "actual_CPU=os.environ.get('SLURM_JOB_ID','NO-SLURM')+'.'+os.environ.get('SLURM_STEP_ID','CPU-recording')")
  needle='def main():\n';assert s.count(needle)==1
  s=s.replace(needle,'''def dependency_paths(text):
 paths=set()
 for line in text.splitlines():
  match=re.match(r'^\\s*(?:\\S+\\s+=>\\s+)?(/\\S+)\\s+\\(',line)
  if match:paths.add(match.group(1))
 return paths
def main():
''')
  s=s.replace("paths=set(re.findall(r'=>\\s+(/\\S+)',unresolved));resolved={p:sha(p)for p in paths}","paths=dependency_paths(unresolved);resolved={p:sha(p)for p in paths}")
  assert 'paths=dependency_paths(unresolved)'in s
  s=s.replace("q['HOST_TUs_compiled']=1", "q['HOST_TUs_compiled']=1;q['compiled_library_SHA256']=sha(library)")
  start=s.index("if B.exists():raise SystemExit('Immutable output exists; no resubmit')")
  tail=s[start:];s=s[:start]+"if __name__=='__main__':\n"+'\n'.join(' '+line for line in tail.splitlines())+'\n'
 if name=='invoke_CPU_once.py':
  s=s.replace('Kimi391-CPU-once-intent','Kimi391-v2-CPU-once-intent').replace('Kimi391-one-HOST-CPU','Kimi391-v2-one-HOST-CPU').replace('Kimi391-CPU-submitted','Kimi391-v2-CPU-submitted')
 if name=='check_CPU.py':
  s=s.replace('from GPU_run import build_payload,dispatch','from GPU_run import build_payload,dispatch\nfrom CPU_build import dependency_paths')
  needle="source=(D/'adapter.cpp').read_text();"
  s=s.replace(needle,"ldd=(D/'original782-ldd-r.log').read_text();paths=dependency_paths(ldd);assert '/refactor/candidates/Kimi-append512-r247/libKimi_append512_r247.so'in paths and '/refactor/candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so'in paths;assert any(p.endswith('libinfiniops.so')for p in paths);passed('actual782_direct_and_arrow_ldd_paths');\n "+needle)
  s=s.replace("assert 'output->narrow({{1,start,64}})->copy_from(r,y)'in source", "assert 'held.clear();std::vector<rt::Tensor>().swap(held);r.syncStream();double wall'in source\n assert 'output->narrow({{1,start,64}})->copy_from(r,y)'in source")
 (N/name).write_text(s,encoding='utf8',newline='\n')
for p in N.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
assert sha(N/'GPU_run.py')==sha(D/'GPU_run.py')and sha(N/'plan.py')==sha(D/'plan.py')
proof=dict(original_CPU='910407.782',original_overall='FAILED1',original_receipt_SHA256=old['CPU-build.json']['SHA256'],compiled_linked_artifact=old['library'],MD_SHA256=old['adapter.d']['SHA256'],DT_NEEDED_SHA256=old['DT-needed.log']['SHA256'],container_ldd_r_log_SHA256=old['container-ldd-r.log']['SHA256'],compile_link_and_ldd_r_succeeded=True,error='Final metadata parser omitted direct absolute DT_NEEDED line; not compile/ABI/numerical failure',v2_changes=['Release heldA1/B8/C8 temporary outputs and vector storage before finaldrain andtimer endpoint','Parse original actualldd direct+arrow paths; replayread-onlyactual782log in CPUglue'],Native_SDK_GPU_objects_changed=False,original_source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'))
(E/'Kimi391-CPU782-compiled-subscope-and-v2-fix.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
print(json.dumps({'prepared_v2':True,'new_HOST_TUs':1,'GPU':0,'original782_unchanged':True,'GPU_driver_plan_bytes_unchanged':True}))
