from pathlib import Path
import json,hashlib,base64,ast,subprocess,sys
B=Path(__file__).resolve().parent;D=B/'gpu-v1';G=B.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
assert all(sha(D/n)==h for n,h in c['files_SHA256'].items()),'CPU737 qualified bytes changed'
data=json.loads((L/'Kimi369-entry-CPU-first-observed.json').read_bytes());data.pop('log_tail',None)
q=data['receipt'];assert q['completed']and q['passed']and q['actual_CPU']=='910407.737'and q['checks']['query_records']==66 and q['checks']['protocol_commands']==159
assert q['source_manifest_SHA256']==sha(D/'CPU_SOURCE_MANIFEST.json')
O=G/'results/bw1000/2026-11-02';(O/'Kimi369-CPU737-whole-query-entry-PASS.json').write_text(json.dumps(data,indent=2)+'\n')
p=subprocess.run([sys.executable,'-B',str(D/'check_launch_CPU.py')],text=True,capture_output=True,check=True)
local=json.loads(p.stdout);assert local['passed']and local['GPU']==0
(O/'Kimi369-launch-once-local-CPU-PASS.json').write_text(json.dumps(local,indent=2)+'\n')
names=list(c['files_SHA256'])+['CPU_SOURCE_MANIFEST.json','idle.py','GPU_inside.sh','controller.py','launch_once.py','check_launch_CPU.py']
for n in names:
 path=D/n
 if path.suffix=='.py':ast.parse(path.read_text())
 if path.suffix=='.sh':assert b'\r'not in path.read_bytes()
old=G/'code/bw1000/Kimi-long-draft-memory-floor-r362b';old_manifest=json.loads((old/'SOURCE_MANIFEST.json').read_bytes())
deps=dict(c['dependencies_SHA256']);deps['/refactor/incoming/Kimi-long-draft-memory-floor-r362b/SOURCE_MANIFEST.json']=sha(old/'SOURCE_MANIFEST.json')
deps.update({'/refactor/incoming/Kimi-long-draft-memory-floor-r362b/'+n:h for n,h in old_manifest['files_SHA256'].items()})
cp='/refactor/candidates/Kimi-local-context-crop-r369-gpu-v1-CPU/CPU-result.json';deps[cp]=data['receipt_SHA256']
claim=dict(GPU_parent=910371,GPU_node='a02r3n15',cards=1,CPUs=8,memory_MiB=65536,GPU_max_s=600,driver_original_epoch_deadline_s=570,CPU_keeper_parent=910407,CPU_keeper_node='a14r2n20',CPU_keeper_CPUs=1,CPU_keeper_memory_MiB=4096,CPU_keeper_max_s=900,target_cards=0,one_attempt=True,profile=False,lengths=[1024,8635,32638],per_context_schedule='2warm OFF/ON then[OFF,ON,ON,OFF]*5',queries=66,append_prompt=83,begin=3,seal=3,retire=3,close=1,total_protocol_commands=159,GuardedCodec_fences_expected=215,feature_bytes=42297*35840*2,query_positions=8,proposals_per_query=7,anchor=163584,context_capacity=32768,strict_checkpoint_and_target_head=True,synthetic_periodic_seed324=True,no_target_reload=True,no_retry=True,no_window_grid=True,owner='/root/kimi_operator_tuning',timing='Whole DraftSession.dispatch query; no TCP wire cost. Append and strictload separate; external numeric D2H and full context hash outside paired query wall.',cleanup='Original Slurm+srun terminal and actual normal runtime/session close; controller post KFD/procs+ports46947/46962 then independent CPU-only node15/keeper check',scope='Synthetic geometry/component only, not target acceptance/Model/online qualification')
m=dict(GPU_GO=True,files_SHA256={n:sha(D/n)for n in names},dependencies_SHA256=deps,CPU_receipt=cp,CPU_receipt_SHA256=data['receipt_SHA256'],actual_CPU=q['actual_CPU'],CPU_source_manifest_SHA256=sha(D/'CPU_SOURCE_MANIFEST.json'),worker_manifest_SHA256=sha(old/'SOURCE_MANIFEST.json'),outer_wrapper_SHA256=old_manifest['outer_wrapper_SHA256'],claim=claim,mathematical_source='Base369CPU736 frozen MF2806 + original pinned0794 method with sole absolute-span argument AST edit; first4 only window4095, fifthglobal/full8draft/cpdp/weights/head/selector unchanged',qualification='CPU736 actualtiny math and CPU737 actual66query lifecycle passed; actual checkpoint GPU numerics/performance and acceptance UNKNOWN')
(D/'GPU_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'GPU_MANIFEST.json')
t=json.loads((G/'coordination/tasks/Kimi-local-context-crop-r369.json').read_bytes());t.update(state='CPU736_737_PASS_ONE_DRAFT_QUERY_GPU_CLAIM',GPU_GO=True,new_GPU_submitted=False,GPU_manifest_SHA256=h,CPU737=dict(actual='910407.737',receipt_SHA256=data['receipt_SHA256'],result='results/bw1000/2026-11-02/Kimi369-CPU737-whole-query-entry-PASS.json'),once_GPU_claim=claim);(G/'coordination/tasks/Kimi-local-context-crop-r369.json').write_text(json.dumps(t,indent=2)+'\n')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in names+['GPU_MANIFEST.json']if n not in c['files_SHA256']and n!='CPU_SOURCE_MANIFEST.json'}
script="""from pathlib import Path
import hashlib,json,base64
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-local-context-crop-r369/gpu-v1'
c=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes());assert hashlib.sha256((D/'CPU_SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__CPUHASH__
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==v for n,v in c['files_SHA256'].items())
for n,b in __PAYLOAD__.items():
 p=D/n;assert not p.exists();p.write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'GPU_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==v for n,v in m['files_SHA256'].items())
print(json.dumps(dict(uploaded=True,GPU_submitted=False,source_manifest_SHA256=__HASH__,CPU737_bytes_unchanged=True)))
""".replace('__CPUHASH__',repr(sha(D/'CPU_SOURCE_MANIFEST.json'))).replace('__HASH__',repr(h)).replace('__PAYLOAD__',repr(payload))
ast.parse(script);(L/'remote369-upload-final.py').write_bytes(script.encode())
print(json.dumps(dict(GPU_manifest_SHA256=h,worker_manifest_SHA256=m['worker_manifest_SHA256'],CPU737_SHA256=data['receipt_SHA256'],GPU_claim=claim,helper_AST_passed=True)))
