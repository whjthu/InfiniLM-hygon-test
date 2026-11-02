from pathlib import Path
import hashlib,json,ast,base64
B=Path(__file__).resolve().parent;D=B/'gpu-v1';G=B.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['GPU_driver.py','window_context.py','check_entry_CPU.py','CPU_main.py','CPU_inside.sh']
for n in names:
 p=D/n
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
old=json.loads((B/'SOURCE_MANIFEST.json').read_bytes());deps=dict(old['dependencies_SHA256'])
deps.update({'/refactor/incoming/Kimi-local-context-crop-r369/'+n:h for n,h in old['files_SHA256'].items()})
deps['/refactor/incoming/Kimi-local-context-crop-r369/SOURCE_MANIFEST.json']=sha(B/'SOURCE_MANIFEST.json')
q=json.loads((L/'Kimi369-CPU-first-observed.json').read_bytes());assert q['receipt']['passed']and q['receipt']['actual_CPU']=='910407.736'
q.pop('log_tail',None)
O=G/'results/bw1000/2026-11-02';(O/'Kimi369-CPU736-dependency-query-PASS.json').write_text(json.dumps(q,indent=2)+'\n')
depCPU='/refactor/candidates/Kimi-local-context-crop-r369/CPU-result.json';deps[depCPU]=q['receipt_SHA256']
m=dict(files_SHA256={n:sha(D/n)for n in names},dependencies_SHA256=deps,GPU_GO=False,GPU=0,CPU_claim=dict(parent=910407,node='a14r2n20',CPUs=4,memory_MiB=8192,hard_s=600,inner_s=570),CPU_scope='Exact GPU_driver.run_plan + actual333DraftSession/privateWindowContext + tiny BF16 originalfactory; 3contexts1024/8635/32638,66queries,83append,begin/seal/retire/close and input/weight metadata guards; no strictcheckpoint/GPU/floor qualification')
(D/'CPU_SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'CPU_SOURCE_MANIFEST.json')
t=json.loads((G/'coordination/tasks/Kimi-local-context-crop-r369.json').read_bytes());t.update(state='CPU736_PASS_FULL_QUERY_ENTRY_CPU_PENDING',CPU736=dict(actual='910407.736',receipt_SHA256=q['receipt_SHA256'],result='results/bw1000/2026-11-02/Kimi369-CPU736-dependency-query-PASS.json'),entry_CPU_manifest_SHA256=h,entry_CPU_claim=m['CPU_claim'],entry_CPU_scope=m['CPU_scope']);(G/'coordination/tasks/Kimi-local-context-crop-r369.json').write_text(json.dumps(t,indent=2)+'\n')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in names+['CPU_SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import json,base64,hashlib,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');B=R/'incoming/Kimi-local-context-crop-r369';D=B/'gpu-v1';assert not D.exists();D.mkdir()
for n,b in __PAYLOAD__.items():(D/n).write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'CPU_SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=Kimi-r369-entry-CPU','--output='+str(R/'logs/Kimi-r369-entry-CPU-%J.log'),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323='Kimi-local-context-crop-r369/gpu-v1/CPU_inside.sh')
with(R/'logs/Kimi-r369-entry-CPU-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_submitted=True,GPU=0,PID=p.pid,label='Kimi-r369-entry-CPU',source_manifest_SHA256=__HASH__)))
""".replace('__PAYLOAD__',repr(payload)).replace('__HASH__',repr(h))
ast.parse(script);(L/'remote369-upload-entry-CPU.py').write_text(script,encoding='utf8')
print(json.dumps(dict(CPU_manifest_SHA256=h,GPU=0,helper_AST_passed=True)))
