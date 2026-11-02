"""Reuse the proven complete-layer367 driver; never modify its published files."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;G=D.parents[2];OLD=D.parent/'GLM-smallM-FFN-feasibility-r367';N=D/'gpu-v1'
N.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,s):
 p=N/n;assert not p.exists(),p;p.write_text(s,encoding='utf8',newline='\n')
def reuse(n):return(OLD/n).read_text(encoding='utf8')
s=reuse('GPU_run.py')
start=s.index('def verify():');end=s.index('def api(path):')
s=s[:start]+'''def verify():
 m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes());assert sha(D/'GPU_SOURCE_MANIFEST.json')==os.environ['GLMSHARED_GPU_SOURCE_SHA256']
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 p=R/'results/GLM-shared-M1-direct-readonly-qualification-v2.json';assert sha(p)==m['qualification_receipt_SHA256'];a=json.loads(p.read_bytes());assert a['completed']and a['passed']and a['qualification_only']and not a['Native_dlopen']and a['new_Slurm_jobs']==a['new_HOST_compiles']==a['new_links']==a['GPU']==0
 assert sha(R/'candidates/GLM-shared-M1-direct-cpu-v1/CPU-build.json')==a['original_CPU']['receipt_SHA256']==m['CPU_receipt_SHA256']
 assert sha(a['library'])==m['candidate_library_SHA256']==a['library_SHA256']
 assert sha(R/'candidates/GLM-PP4-registry-repair-r184b/libGLM_PP4_registry_r184b.so')==m['baseline_Native_SHA256']==a['parent_Native_SHA256']
 for p,h in a['all_objects_SHA256'].items():assert sha(p)==h,p
 for p,h in a['actual_project_library_SHA256'].items():assert sha(p)==h,p
 for p,h in m['entry_dependency_SHA256'].items():assert sha(p)==h,p
 return a,m
'''+s[end:]
s=s.replace("for name in ['modern_glm_vector_policy_r8540','modern_glm_down_row2_policy_r9284','modern_glm_M1_gate_policy_r367']:getattr(lib,name).argtypes=[I,I]", "for name in ['modern_glm_vector_policy_r8540','modern_glm_down_row2_policy_r9284','glm_shared_M1_mode_v1']:getattr(lib,name).argtypes=[I,I]")
s=s.replace("lib.modern_glm_M1_gate_get_r367.argtypes=[I];lib.modern_glm_M1_gate_counts_r367.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]", "lib.glm_shared_M1_counts_v1.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]")
s=s.replace("assert lib.modern_glm_M1_gate_get_r367(rank)==0", "check(lib.glm_shared_M1_mode_v1(rank,0));assert lib.glm_shared_M1_mode_v1(rank,2)==1 and lib.glm_shared_M1_mode_v1(-1,0)==1 and lib.glm_shared_M1_mode_v1(8,0)==1")
s=s.replace("'modern_glm_M1_gate_policy_r367','glm_vec_gate_r8528'", "'glm_shared_M1_mode_v1','glm_shared_M1_counts_v1','glm_routed_gate_r19','glm_routed_down_r19','glm_vec_gate_r8528'")
s=s.replace('modern_glm_M1_gate_policy_r367','glm_shared_M1_mode_v1').replace("'modern_glm_M1_gate_counts_r367',4", "'glm_shared_M1_counts_v1',5")
s=s.replace("assert delta==[mode,1-mode,0,0]and vd==[mode,1-mode,0,1,0,mode,0,0],(delta,vd)", "agree(delta==[1,mode,mode,mode,0]and vd==[0,1,0,1,0,0,0,0]);assert delta==[1,mode,mode,mode,0]and vd==[0,1,0,1,0,0,0,0],(delta,vd)")
s=s.replace('GLM367','GLMSHARED').replace('367001','387001').replace("only M1 gate/up provider changes", "only singleton shared gate/up/down provider changes; routed8 and TP unchanged; OFF retains original math with counters")
s=s.replace("['measurement_calls']==48", "['measurement_calls']==96")
s=s.replace('M1_counts_delta=delta','shared_M1_counts_delta=delta')
assert 'r367'not in s and "'glm_shared_M1_counts_v1',5"in s
write('GPU_run.py',s)
write('plan.py', '''FIXTURES=2
ORDERS=((0,1,1,0),)*4
REPS=3
def records(fixture):
 assert 0<=fixture<FIXTURES
 yield from(dict(mode=mode,warm=True,pair=-1,arm=mode,rep=0)for mode in [0,1])
 for pair,order in enumerate(ORDERS):
  for arm,mode in enumerate(order):
   for rep in range(REPS):yield dict(mode=mode,warm=False,pair=pair,arm=arm,rep=rep)
''')
for n in ['GPU_inside.sh','resource_probe.py','controller.py','prepare_GPU_submit.py','audit_terminal.py']:
 s=reuse(n).replace('GLM-smallM-FFN-feasibility-r367','GLM-shared-M1-direct/gpu-v1').replace('GLM-r367-M1-gate-layer','GLM-shared-M1-direct-layer-v1').replace('a02r3n03','a02r3n16').replace('GLM367','GLMSHARED')
 s=s.replace("['measurement_calls']==48", "['measurement_calls']==96").replace("['trials'])==48", "['trials'])==96").replace("in a['trials']})==48", "in a['trials']})==96")
 if n=='controller.py':
  s=s.replace("assert sha(Path(root)/'candidates/GLM-smallM-FFN-r367/CPU-build.json')==m['CPU_receipt_SHA256']", "assert sha(Path(root)/'candidates/GLM-shared-M1-direct-cpu-v1/CPU-build.json')==m['CPU_receipt_SHA256'];assert sha(Path(root)/'results/GLM-shared-M1-direct-readonly-qualification-v2.json')==m['qualification_receipt_SHA256']")
 if n=='prepare_GPU_submit.py':
  s=s.replace("assert D.is_dir()and not", "assert not D.exists()and not").replace("for n,b in FILE_PAYLOAD.items():", "D.mkdir(parents=True)\nfor n,b in FILE_PAYLOAD.items():")
  s=s.replace("original_CPU='910407.726'", "original_CPU='910407.774 FAILED1; compile/link successful; readonly qualifier PASS'").replace('Onefreshn03','Onefreshn16')
  s=s.replace("p=D.parents[3]/'remote367-GPU-submit.py'", "p=D.parents[4]/'remote-shared-M1-GPU-submit.py'")
 if n=='audit_terminal.py':
  s=s.replace("r['M1_counts_delta']==[mode,1-mode,0,0]and r['vector_counts_delta']==[mode,1-mode,0,1,0,mode,0,0]", "r['shared_M1_counts_delta']==[1,mode,mode,mode,0]and r['vector_counts_delta']==[0,1,0,1,0,0,0,0]")
  s=s.replace('all(len(v)==3 for v in grouped.values())','all(len(v)==6 for v in grouped.values())').replace("CPU_actual='910407.726'", "CPU_actual='910407.774 FAILED1 retained; readonly qualifier PASS'")
 write(n,s)
for p in N.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
assert all(b'\r'not in p.read_bytes()for p in N.glob('*.sh'))
print(json.dumps({'GPU_source_files':sorted(p.name for p in N.iterdir()),'old_source_unchanged':True,'GPU':0,'Slurm':0}))
