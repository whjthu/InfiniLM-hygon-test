from pathlib import Path
import json,hashlib,ast,subprocess,sys,base64
B=Path(__file__).resolve().parent;D=B/'device-v1';G=B.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cpu=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in cpu['files_SHA256'].items()),'CPU759 qualified source changed'
data=json.loads((L/'Kimi376-device-CPU-first-observed.json').read_bytes());data.pop('log_tail',None);data.pop('last_sequence',None);q=data['receipt'];c=q['checks']
assert q['completed']and q['passed']and q['actual_CPU']=='910407.759'and q['source_manifest_SHA256']==sha(D/'CPU_SOURCE_MANIFEST.json')
for k,v in dict(sequences=18,warm_queries=48,formal_queries=96,commands=840,prompt_appends=498,synthetic_committed_appends=126,main_codec_fences=912).items():assert c[k]==v
assert c['partial_numeric_failure_saved_all_fields']and c['all_normal_context_close']and not c['GPU']
O=G/'results/bw1000/2026-11-02';(O/'Kimi376-CPU759-exact-whole-plan-PASS.json').write_text(json.dumps(data,indent=2)+'\n')
p=subprocess.run([sys.executable,'-B',str(D/'check_launch_CPU.py')],capture_output=True,text=True,check=True);local=json.loads(p.stdout);assert local['passed']and local['GPU']==0
(O/'Kimi376-device-launch-local-CPU-PASS.json').write_text(json.dumps(local,indent=2)+'\n')
names=list(cpu['files_SHA256'])+['CPU_SOURCE_MANIFEST.json','controller.py','idle.py','GPU_inside.sh','launch_once.py','check_launch_CPU.py']
for n in names:
 path=D/n
 if path.suffix=='.py':ast.parse(path.read_text())
 if path.suffix=='.sh':assert b'\r'not in path.read_bytes()
old=json.loads((B.parent/'Kimi-local-window-first-divergence-r371/GPU_MANIFEST.json').read_bytes())
deps=dict(cpu['dependencies_SHA256']);cp='/refactor/candidates/Kimi-preRoPE-prefix-cache-r376-device-v1-CPU/CPU-result.json';deps[cp]=data['receipt_SHA256']
plan=json.loads((D/'PLAN.json').read_bytes())
claim=dict(owner='/root/kimi_operator_tuning',GPU_parent=910371,GPU_node='a02r3n15',cards=1,CPUs=8,memory_MiB=65536,GPU_max_s=600,driver_original_epoch_deadline_s=570,CPU_keeper_parent=910407,CPU_keeper_node='a14r2n20',CPU_keeper_CPUs=1,CPU_keeper_memory_MiB=4096,CPU_keeper_max_s=900,target_cards=0,one_attempt=True,profile=False,strict_model_and_target_head_loads=1,lengths=plan['geometries'],freshContext_sequences=18,queries_per_sequence=8,warm_queries=48,formal_queries=96,total_queries=144,prompt_append=498,synthetic_committed_append8=126,begin=18,seal=18,retire=18,close=18,total_protocol_commands=840,original_GuardedCodec_fences=912,per_geometry_order=plan['per_geometry_sequence_order'],fixture='Same CPU periodic512 BF16 seed324/raw35840; acceptedfeatures continueperiodic positions, explicitsyntheticcommitproofs not actualtargetacceptance',thresholds=plan['numeric'],timing=plan['timing'],stage_scope=plan['stage_scope'],no_target_reload=True,no_blind_retry=True,no_window_or_allocator_grid=True,decision='Show allpergeometry formalABBA continuouscheckedwall, completequery+acceptedappend dispatchsum, OFFendpointdrift, prefix/warm/firstprocess-shapecall/load/retireclose/cardcost. Retain onlyclearwholesequence netbeyonddrift; no observerfree/Model/acceptance conclusion.',cleanup='OriginalactualGPU/keeper/srun terminal andall18Contextclose, Runtimeclose; original andindependent CPU KFD/PID/ports46947/46962 checks')
m=dict(GPU_GO=True,files_SHA256={n:sha(D/n)for n in names},dependencies_SHA256=deps,CPU_receipt=cp,CPU_receipt_SHA256=data['receipt_SHA256'],actual_CPU='910407.759',CPU_source_manifest_SHA256=sha(D/'CPU_SOURCE_MANIFEST.json'),core_CPU752_SHA256='919c8faf461b4536a509ee76c2f52f454d6bbfac81389454bc4f1a14c2dc7bda',worker_manifest_SHA256=old['worker_manifest_SHA256'],outer_wrapper_SHA256=old['outer_wrapper_SHA256'],claim=claim,scope='PreRoPE twoLinear cache only, originalattention/fullmask/RoPE/SDPA/head/selector andfrozenCPU752core unchanged. CPU759 sharedactualtiny144query plan andfailure-saved-fields passed; realcheckpoint/GPUperformance/numerics unknown.')
(D/'GPU_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'GPU_MANIFEST.json')
tp=G/'coordination/tasks/Kimi-preRoPE-prefix-cache-r376.json';t=json.loads(tp.read_bytes());t.update(state='CPU752_759_PASS_ONE_DRAFT_18SEQUENCE_GPU_CLAIM',GPU_GO=True,new_GPU_submitted=False,device_GPU_manifest_SHA256=h,CPU759=dict(actual='910407.759',receipt_SHA256=data['receipt_SHA256'],result='results/bw1000/2026-11-02/Kimi376-CPU759-exact-whole-plan-PASS.json'),once_GPU_claim=claim);tp.write_text(json.dumps(t,indent=2)+'\n')
new=[n for n in names if n not in cpu['files_SHA256']and n!='CPU_SOURCE_MANIFEST.json']+['GPU_MANIFEST.json'];payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in new}
script="""from pathlib import Path
import json,hashlib,base64
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-preRoPE-prefix-cache-r376/device-v1'
assert hashlib.sha256((D/'CPU_SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__CPUHASH__
c=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in c['files_SHA256'].items())
for n,b in __PAYLOAD__.items():
 p=D/n;assert not p.exists();p.write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'GPU_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
print(json.dumps(dict(uploaded=True,GPU_submitted=False,GPU_manifest_SHA256=__HASH__,CPU759_source_unchanged=True)))
""".replace('__CPUHASH__',repr(sha(D/'CPU_SOURCE_MANIFEST.json'))).replace('__HASH__',repr(h)).replace('__PAYLOAD__',repr(payload))
ast.parse(script);(L/'remote376-device-upload-final.py').write_bytes(script.encode())
print(json.dumps(dict(GPU_manifest_SHA256=h,CPU759_SHA256=data['receipt_SHA256'],claim=claim,helper_AST_passed=True)))
